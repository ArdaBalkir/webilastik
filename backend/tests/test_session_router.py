from __future__ import annotations

import asyncio
import threading
import time
import unittest

from backend.session_router import (
    AdmissionRejected,
    ClassifierConflict,
    ClassifierForbidden,
    ClassifierGone,
    ClassifierUnknown,
    FairWorkerScheduler,
    InMemoryRegistry,
    RouterConfig,
    SessionRouter,
    WorkKind,
    WorkerState,
)


def config(**overrides):
    values = {
        "worker_count": 1,
        "predict_slots_per_worker": 1,
        "train_slots_per_worker": 1,
        "predict_queue_limit": 8,
        "train_queue_limit": 8,
        "per_user_predict_limit": 8,
        "per_user_train_limit": 3,
        "model_ttl_seconds": 30,
        "cleanup_interval_seconds": 60,
        "shutdown_timeout_seconds": 0.2,
        "request_timeout_seconds": 3,
        "prediction_cache_bytes": 0,
        "prediction_burst_before_training": 2,
    }
    values.update(overrides)
    return RouterConfig(**values)


async def wait_for_event(event: threading.Event) -> None:
    for _ in range(300):
        if event.is_set():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("worker function did not start")


class RegistryTests(unittest.TestCase):
    def test_authorization_lifecycle_and_generation(self):
        registry = InMemoryRegistry(ttl_seconds=30)
        record = registry.allocate("alice", "worker-1", {"filters": ["gaussian"]})

        with self.assertRaises(ClassifierConflict):
            registry.resolve(record.classifier_id, "alice")

        registry.publish_ready(record.classifier_id)
        resolved = registry.resolve(record.classifier_id, "alice", record.generation)
        self.assertEqual(resolved.worker_id, "worker-1")

        with self.assertRaises(ClassifierForbidden):
            registry.resolve(record.classifier_id, "bob")
        with self.assertRaises(ClassifierConflict):
            registry.resolve(record.classifier_id, "alice", record.generation + 1)
        with self.assertRaises(ClassifierUnknown):
            registry.resolve("never-known", "alice")


class RouterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        router = getattr(self, "router", None)
        if router is not None:
            await router.close()

    async def test_classifier_worker_affinity(self):
        self.router = SessionRouter(
            config(worker_count=2, predict_slots_per_worker=2)
        )
        model = object()
        trained = await self.router.train("alice", {}, lambda: model)

        record = self.router.authorize(trained.classifier_id, "alice")
        self.assertEqual(record.worker_id, trained.worker_id)
        self.assertTrue(
            self.router.workers[record.worker_id].has_model(trained.classifier_id)
        )
        observed = await self.router.run_model_task(
            trained.classifier_id,
            "alice",
            WorkKind.PREDICTION,
            id,
        )
        self.assertEqual(observed, id(model))

    async def test_retraining_race_keeps_ready_model_and_rejects_stale_publish(self):
        self.router = SessionRouter(
            config(worker_count=2, predict_slots_per_worker=2)
        )
        original = await self.router.train("alice", {}, lambda: "original")

        slow_started = threading.Event()
        release_slow = threading.Event()

        def slow_train():
            slow_started.set()
            release_slow.wait(timeout=2)
            return "stale"

        stale_task = asyncio.create_task(
            self.router.train("alice", {"version": 2}, slow_train)
        )
        await wait_for_event(slow_started)

        # The prior ready generation remains usable while replacement trains.
        still_ready = await self.router.run_model_task(
            original.classifier_id,
            "alice",
            WorkKind.PREDICTION,
            lambda model: model,
        )
        self.assertEqual(still_ready, "original")

        newest = await self.router.train("alice", {"version": 3}, lambda: "newest")
        release_slow.set()
        with self.assertRaises(ClassifierConflict):
            await stale_task

        value = await self.router.run_model_task(
            newest.classifier_id,
            "alice",
            WorkKind.PREDICTION,
            lambda model: model,
        )
        self.assertEqual(value, "newest")
        with self.assertRaises(ClassifierGone):
            self.router.authorize(original.classifier_id, "alice")

    async def test_queue_capacity_returns_explicit_backpressure(self):
        self.router = SessionRouter(
            config(predict_queue_limit=1, per_user_predict_limit=10)
        )
        trained = await self.router.train("alice", {}, lambda: object())
        started = threading.Event()
        release = threading.Event()

        def blocking(_model):
            started.set()
            release.wait(timeout=2)
            return "first"

        first = asyncio.create_task(
            self.router.run_model_task(
                trained.classifier_id, "alice", WorkKind.PREDICTION, blocking
            )
        )
        await wait_for_event(started)
        second = asyncio.create_task(
            self.router.run_model_task(
                trained.classifier_id,
                "alice",
                WorkKind.PREDICTION,
                lambda _model: "second",
            )
        )
        await asyncio.sleep(0.02)

        with self.assertRaises(AdmissionRejected) as raised:
            await self.router.run_model_task(
                trained.classifier_id,
                "alice",
                WorkKind.PREDICTION,
                lambda _model: "third",
            )
        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(raised.exception.retry_after, 1)

        release.set()
        self.assertEqual(await first, "first")
        self.assertEqual(await second, "second")

    async def test_per_user_limit_returns_429(self):
        self.router = SessionRouter(
            config(predict_queue_limit=4, per_user_predict_limit=1)
        )
        trained = await self.router.train("alice", {}, lambda: object())
        started = threading.Event()
        release = threading.Event()

        first = asyncio.create_task(
            self.router.run_model_task(
                trained.classifier_id,
                "alice",
                WorkKind.PREDICTION,
                lambda _model: (started.set(), release.wait(timeout=2))[1],
            )
        )
        await wait_for_event(started)
        with self.assertRaises(AdmissionRejected) as raised:
            await self.router.run_model_task(
                trained.classifier_id,
                "alice",
                WorkKind.PREDICTION,
                lambda _model: None,
            )
        self.assertEqual(raised.exception.status_code, 429)
        release.set()
        await first

    async def test_expiry_removes_model_and_leaves_deterministic_tombstone(self):
        self.router = SessionRouter(config(model_ttl_seconds=0.02))
        trained = await self.router.train("alice", {}, lambda: object())
        await asyncio.sleep(0.03)
        expired = await self.router.expire_once()

        self.assertEqual([item.classifier_id for item in expired], [trained.classifier_id])
        self.assertFalse(
            self.router.workers[trained.worker_id].has_model(trained.classifier_id)
        )
        with self.assertRaises(ClassifierGone) as raised:
            self.router.authorize(trained.classifier_id, "alice")
        self.assertIn("expired", raised.exception.detail.lower())

    async def test_draining_preserves_affinity_but_stops_new_assignments(self):
        self.router = SessionRouter(config(worker_count=2))
        first = await self.router.train("alice", {}, lambda: "model-a")
        await self.router.drain_worker(first.worker_id)

        self.assertEqual(
            await self.router.run_model_task(
                first.classifier_id,
                "alice",
                WorkKind.PREDICTION,
                lambda model: model,
            ),
            "model-a",
        )
        second = await self.router.train("bob", {}, lambda: "model-b")
        self.assertNotEqual(second.worker_id, first.worker_id)
        self.assertEqual(
            self.router.workers[first.worker_id].scheduler.state,
            WorkerState.DRAINING,
        )

    async def test_worker_failure_marks_classifier_lost(self):
        self.router = SessionRouter(config())
        trained = await self.router.train("alice", {}, lambda: object())
        await self.router.mark_worker_unhealthy(trained.worker_id, "simulated crash")

        with self.assertRaises(ClassifierGone) as raised:
            self.router.authorize(trained.classifier_id, "alice")
        self.assertIn("retraining", raised.exception.detail.lower())
        health = self.router.health()
        self.assertEqual(health["worker_summary"]["unhealthy"], 1)
        self.assertEqual(health["counters"]["classifiers_lost"], 1)

    async def test_training_idempotency_replays_and_rejects_body_mismatch(self):
        self.router = SessionRouter(config())
        calls = 0

        def train_once():
            nonlocal calls
            calls += 1
            return object()

        first = await self.router.train(
            "alice", {}, train_once, idempotency_key="same", fingerprint="body-a"
        )
        replay = await self.router.train(
            "alice", {}, train_once, idempotency_key="same", fingerprint="body-a"
        )
        self.assertEqual(first.classifier_id, replay.classifier_id)
        self.assertEqual(calls, 1)

        with self.assertRaises(ClassifierConflict):
            await self.router.train(
                "alice", {}, train_once, idempotency_key="same", fingerprint="body-b"
            )


class SchedulerFairnessTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        scheduler = getattr(self, "scheduler", None)
        if scheduler is not None:
            await scheduler.close(0.2)

    async def test_round_robin_prevents_one_user_from_draining_queue(self):
        self.scheduler = FairWorkerScheduler("test", config())
        order = []
        started = threading.Event()
        release = threading.Event()

        def first():
            order.append("a1")
            started.set()
            release.wait(timeout=2)

        tasks = [
            asyncio.create_task(self.scheduler.submit(WorkKind.PREDICTION, "a", first))
        ]
        await wait_for_event(started)
        for user, label in [("a", "a2"), ("a", "a3"), ("b", "b1")]:
            tasks.append(
                asyncio.create_task(
                    self.scheduler.submit(
                        WorkKind.PREDICTION,
                        user,
                        lambda current=label: order.append(current),
                    )
                )
            )
        await asyncio.sleep(0.02)
        release.set()
        await asyncio.gather(*tasks)

        self.assertLess(order.index("b1"), order.index("a3"))

    async def test_prediction_priority_has_training_anti_starvation(self):
        self.scheduler = FairWorkerScheduler("test", config())
        order = []
        started = threading.Event()
        release = threading.Event()

        def blocker():
            order.append("blocker")
            started.set()
            release.wait(timeout=2)

        tasks = [
            asyncio.create_task(
                self.scheduler.submit(WorkKind.PREDICTION, "a", blocker)
            )
        ]
        await wait_for_event(started)
        tasks.extend(
            [
                asyncio.create_task(
                    self.scheduler.submit(
                        WorkKind.TRAINING, "trainer", lambda: order.append("train")
                    )
                ),
                asyncio.create_task(
                    self.scheduler.submit(
                        WorkKind.PREDICTION, "b", lambda: order.append("p1")
                    )
                ),
                asyncio.create_task(
                    self.scheduler.submit(
                        WorkKind.PREDICTION, "c", lambda: order.append("p2")
                    )
                ),
            ]
        )
        await asyncio.sleep(0.02)
        release.set()
        await asyncio.gather(*tasks)

        self.assertEqual(order[:3], ["blocker", "p1", "train"])
        self.assertIn("p2", order)


if __name__ == "__main__":
    unittest.main()
