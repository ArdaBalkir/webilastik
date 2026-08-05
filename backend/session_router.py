"""Single-process session registry and bounded interactive worker router.

The implementation in this module is intentionally scoped to one FastAPI
process on one VM.  The registry interface makes the persistence boundary
explicit, but the supplied registry and workers are process-local.  A shared,
transactional registry and remotely addressable workers are required before
running more than one router process.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import pathlib
import threading
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, Generic, Mapping, Optional, Protocol, TypeVar


logger = logging.getLogger(__name__)
T = TypeVar("T")


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _positive_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True)
class RouterConfig:
    worker_count: int = 2
    predict_slots_per_worker: int = 2
    train_slots_per_worker: int = 1
    predict_queue_limit: int = 32
    train_queue_limit: int = 8
    per_user_predict_limit: int = 8
    per_user_train_limit: int = 2
    model_ttl_seconds: float = 30 * 60
    cleanup_interval_seconds: float = 30
    shutdown_timeout_seconds: float = 20
    request_timeout_seconds: float = 300
    worker_memory_high_watermark_bytes: int = 0
    prediction_cache_bytes: int = 32 * 1024 * 1024
    prediction_burst_before_training: int = 4
    idempotency_ttl_seconds: float = 10 * 60
    worker_weights: tuple[float, ...] = ()

    @classmethod
    def from_env(cls) -> "RouterConfig":
        worker_count = _positive_int("INTERACTIVE_WORKER_COUNT", 2)
        weights_raw = os.environ.get("INTERACTIVE_WORKER_WEIGHTS", "")
        weights: tuple[float, ...] = ()
        if weights_raw.strip():
            parsed = tuple(float(part.strip()) for part in weights_raw.split(","))
            if len(parsed) != worker_count or any(weight <= 0 for weight in parsed):
                raise ValueError(
                    "INTERACTIVE_WORKER_WEIGHTS must contain one positive weight per worker"
                )
            weights = parsed
        watermark_mb = float(os.environ.get("WORKER_MEMORY_HIGH_WATERMARK_MB", "0"))
        if watermark_mb < 0:
            raise ValueError("WORKER_MEMORY_HIGH_WATERMARK_MB cannot be negative")
        return cls(
            worker_count=worker_count,
            predict_slots_per_worker=_positive_int("PREDICT_SLOTS_PER_WORKER", 2),
            train_slots_per_worker=_positive_int("TRAIN_SLOTS_PER_WORKER", 1),
            predict_queue_limit=_positive_int("PREDICT_QUEUE_LIMIT", 32),
            train_queue_limit=_positive_int("TRAIN_QUEUE_LIMIT", 8),
            per_user_predict_limit=_positive_int("PER_USER_PREDICT_LIMIT", 8),
            per_user_train_limit=_positive_int("PER_USER_TRAIN_LIMIT", 2),
            model_ttl_seconds=_positive_float("MODEL_TTL_SECONDS", 30 * 60),
            cleanup_interval_seconds=_positive_float("CLEANUP_INTERVAL_SECONDS", 30),
            shutdown_timeout_seconds=_positive_float("WORKER_SHUTDOWN_TIMEOUT_SECONDS", 20),
            request_timeout_seconds=_positive_float("REQUEST_TIMEOUT_SECONDS", 300),
            worker_memory_high_watermark_bytes=int(watermark_mb * 1024 * 1024),
            prediction_cache_bytes=max(
                0, int(float(os.environ.get("PREDICTION_CACHE_MB", "32")) * 1024 * 1024)
            ),
            prediction_burst_before_training=_positive_int(
                "PREDICTION_BURST_BEFORE_TRAINING", 4
            ),
            idempotency_ttl_seconds=_positive_float("IDEMPOTENCY_TTL_SECONDS", 10 * 60),
            worker_weights=weights,
        )


class ClassifierState(str, Enum):
    ALLOCATING = "allocating"
    TRAINING = "training"
    READY = "ready"
    DRAINING = "draining"
    EXPIRED = "expired"
    FAILED = "failed"
    LOST = "lost"
    DELETED = "deleted"


class WorkerState(str, Enum):
    HEALTHY = "healthy"
    DRAINING = "draining"
    UNHEALTHY = "unhealthy"
    STOPPED = "stopped"


class WorkKind(str, Enum):
    PREDICTION = "prediction"
    TRAINING = "training"


class SessionRouterError(Exception):
    status_code = 500
    retry_after: Optional[int] = None

    def __init__(self, detail: str, *, retry_after: Optional[int] = None):
        super().__init__(detail)
        self.detail = detail
        if retry_after is not None:
            self.retry_after = retry_after


class ClassifierUnknown(SessionRouterError):
    status_code = 404


class ClassifierForbidden(SessionRouterError):
    status_code = 403


class ClassifierConflict(SessionRouterError):
    status_code = 409


class ClassifierGone(SessionRouterError):
    status_code = 410


class AdmissionRejected(SessionRouterError):
    def __init__(self, detail: str, status_code: int, reason: str):
        super().__init__(detail, retry_after=1)
        self.status_code = status_code
        self.reason = reason


@dataclass
class ClassifierRecord:
    classifier_id: str
    user_id: str
    worker_id: str
    generation: int
    state: ClassifierState
    created_at: float
    last_used_at: float
    expires_at: float
    feature_config: Mapping[str, Any] = field(default_factory=dict)
    snapshot_location: Optional[str] = None
    failure_reason: Optional[str] = None

    def public_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["state"] = self.state.value
        return result


class Registry(Protocol):
    """Atomic classifier metadata operations required by a router."""

    def allocate(
        self, user_id: str, worker_id: str, feature_config: Mapping[str, Any]
    ) -> ClassifierRecord: ...

    def resolve(
        self, classifier_id: str, user_id: str, generation: Optional[int] = None
    ) -> ClassifierRecord: ...

    def publish_ready(self, classifier_id: str) -> Optional[ClassifierRecord]: ...

    def mark_failed(self, classifier_id: str, reason: str) -> None: ...

    def expire_due(self, now: Optional[float] = None) -> list[ClassifierRecord]: ...

    def mark_worker_lost(self, worker_id: str, reason: str) -> list[ClassifierRecord]: ...

    def mark_worker_draining(self, worker_id: str) -> list[ClassifierRecord]: ...

    def snapshot(self) -> dict[str, Any]: ...


class InMemoryRegistry:
    """Thread-safe, authoritative registry for exactly one router process."""

    def __init__(self, ttl_seconds: float, tombstone_seconds: float = 3600):
        self._ttl_seconds = ttl_seconds
        self._tombstone_seconds = max(tombstone_seconds, ttl_seconds)
        self._records: Dict[str, ClassifierRecord] = {}
        self._tombstones: Dict[str, tuple[ClassifierRecord, float]] = {}
        self._latest_generation: Dict[str, int] = defaultdict(int)
        self._ready_by_user: Dict[str, str] = {}
        self._lock = threading.RLock()
        self._counters: Dict[str, int] = defaultdict(int)

    def allocate(
        self, user_id: str, worker_id: str, feature_config: Mapping[str, Any]
    ) -> ClassifierRecord:
        now = time.time()
        with self._lock:
            generation = self._latest_generation[user_id] + 1
            self._latest_generation[user_id] = generation
            record = ClassifierRecord(
                classifier_id=str(uuid.uuid4()),
                user_id=user_id,
                worker_id=worker_id,
                generation=generation,
                state=ClassifierState.TRAINING,
                created_at=now,
                last_used_at=now,
                expires_at=now + self._ttl_seconds,
                feature_config=dict(feature_config),
            )
            self._records[record.classifier_id] = record
            self._counters["allocated"] += 1
            return record

    def _terminal_detail(self, record: ClassifierRecord) -> str:
        if record.state == ClassifierState.LOST:
            return "Classifier model was lost with its worker; retraining is required"
        if record.state == ClassifierState.EXPIRED:
            return "Classifier expired; retraining is required"
        if record.state == ClassifierState.DELETED:
            return "Classifier was replaced or deleted; retraining is required"
        return record.failure_reason or "Classifier is no longer available"

    def resolve(
        self, classifier_id: str, user_id: str, generation: Optional[int] = None
    ) -> ClassifierRecord:
        now = time.time()
        with self._lock:
            record = self._records.get(classifier_id)
            if record is None:
                tombstone = self._tombstones.get(classifier_id)
                if tombstone is None:
                    raise ClassifierUnknown("Classifier not found")
                record = tombstone[0]
            if record.user_id != user_id:
                raise ClassifierForbidden("Classifier belongs to another user")
            if record.state in {
                ClassifierState.EXPIRED,
                ClassifierState.FAILED,
                ClassifierState.LOST,
                ClassifierState.DELETED,
            }:
                raise ClassifierGone(self._terminal_detail(record))
            if record.state in {ClassifierState.ALLOCATING, ClassifierState.TRAINING}:
                raise ClassifierConflict("Classifier is still training")
            if generation is not None and generation != record.generation:
                raise ClassifierConflict(
                    f"Classifier generation mismatch: current generation is {record.generation}"
                )
            record.last_used_at = now
            record.expires_at = now + self._ttl_seconds
            return record

    def publish_ready(self, classifier_id: str) -> Optional[ClassifierRecord]:
        now = time.time()
        with self._lock:
            record = self._records.get(classifier_id)
            if record is None:
                raise ClassifierGone("Classifier allocation no longer exists")
            if self._latest_generation[record.user_id] != record.generation:
                self._terminalize(record, ClassifierState.DELETED, "Superseded training generation")
                self._counters["stale_training"] += 1
                raise ClassifierConflict("Training result was superseded by a newer request")

            previous: Optional[ClassifierRecord] = None
            previous_id = self._ready_by_user.get(record.user_id)
            if previous_id and previous_id != classifier_id:
                previous = self._records.get(previous_id)
                if previous is not None:
                    self._terminalize(
                        previous, ClassifierState.DELETED, "Replaced by a newer generation"
                    )

            record.state = ClassifierState.READY
            record.last_used_at = now
            record.expires_at = now + self._ttl_seconds
            self._ready_by_user[record.user_id] = classifier_id
            self._counters["ready"] += 1
            return previous

    def _terminalize(
        self, record: ClassifierRecord, state: ClassifierState, reason: Optional[str]
    ) -> None:
        self._records.pop(record.classifier_id, None)
        record.state = state
        record.failure_reason = reason
        if self._ready_by_user.get(record.user_id) == record.classifier_id:
            self._ready_by_user.pop(record.user_id, None)
        self._tombstones[record.classifier_id] = (
            record,
            time.time() + self._tombstone_seconds,
        )

    def mark_failed(self, classifier_id: str, reason: str) -> None:
        with self._lock:
            record = self._records.get(classifier_id)
            if record is not None:
                self._terminalize(record, ClassifierState.FAILED, reason)
                self._counters["failed"] += 1

    def expire_due(self, now: Optional[float] = None) -> list[ClassifierRecord]:
        when = time.time() if now is None else now
        expired: list[ClassifierRecord] = []
        with self._lock:
            for record in list(self._records.values()):
                if record.state in {ClassifierState.READY, ClassifierState.DRAINING} and record.expires_at <= when:
                    self._terminalize(record, ClassifierState.EXPIRED, "TTL expired")
                    expired.append(record)
            for classifier_id, (_, remove_at) in list(self._tombstones.items()):
                if remove_at <= when:
                    self._tombstones.pop(classifier_id, None)
            self._counters["expired"] += len(expired)
        return expired

    def mark_worker_lost(self, worker_id: str, reason: str) -> list[ClassifierRecord]:
        lost: list[ClassifierRecord] = []
        with self._lock:
            for record in list(self._records.values()):
                if record.worker_id == worker_id:
                    self._terminalize(record, ClassifierState.LOST, reason)
                    lost.append(record)
            self._counters["lost"] += len(lost)
        return lost

    def mark_worker_draining(self, worker_id: str) -> list[ClassifierRecord]:
        draining: list[ClassifierRecord] = []
        with self._lock:
            for record in self._records.values():
                if record.worker_id == worker_id and record.state == ClassifierState.READY:
                    record.state = ClassifierState.DRAINING
                    draining.append(record)
        return draining

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            states: Dict[str, int] = defaultdict(int)
            for record in self._records.values():
                states[record.state.value] += 1
            return {
                "implementation": "in_memory_single_process",
                "active_classifiers": len(self._records),
                "states": dict(states),
                "tombstones": len(self._tombstones),
                "counters": dict(self._counters),
            }


@dataclass
class WorkItem(Generic[T]):
    kind: WorkKind
    user_id: str
    fn: Callable[[], T]
    future: asyncio.Future[T]
    enqueued_at: float
    classifier_id: Optional[str] = None


class FairWorkerScheduler:
    """Bounded, prediction-first scheduler with per-user round robin."""

    def __init__(self, worker_id: str, config: RouterConfig):
        self.worker_id = worker_id
        self._config = config
        self._max_active = max(
            config.predict_slots_per_worker, config.train_slots_per_worker
        )
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_active, thread_name_prefix=f"interactive-{worker_id}"
        )
        self._queues: dict[WorkKind, OrderedDict[str, Deque[WorkItem[Any]]]] = {
            WorkKind.PREDICTION: OrderedDict(),
            WorkKind.TRAINING: OrderedDict(),
        }
        self._queue_depth: Dict[WorkKind, int] = defaultdict(int)
        self._active: Dict[WorkKind, int] = defaultdict(int)
        self._active_by_user: Dict[tuple[WorkKind, str], int] = defaultdict(int)
        self._lock = asyncio.Lock()
        self._state = WorkerState.HEALTHY
        self._last_seen = time.time()
        self._prediction_burst = 0
        self._counters: Dict[str, int] = defaultdict(int)
        self._queue_wait_seconds: Dict[WorkKind, list[float]] = defaultdict(list)
        self._compute_seconds: Dict[WorkKind, list[float]] = defaultdict(list)

    @property
    def state(self) -> WorkerState:
        return self._state

    def load_score(self, weight: float) -> float:
        queued = sum(self._queue_depth.values())
        active = sum(self._active.values())
        return (queued + active) / weight

    def can_accept_training(self) -> bool:
        return (
            self._state == WorkerState.HEALTHY
            and self._queue_depth[WorkKind.TRAINING] < self._config.train_queue_limit
        )

    async def submit(
        self,
        kind: WorkKind,
        user_id: str,
        fn: Callable[[], T],
        classifier_id: Optional[str] = None,
        allow_draining: bool = False,
    ) -> T:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[T] = loop.create_future()
        item = WorkItem(kind, user_id, fn, future, time.monotonic(), classifier_id)
        async with self._lock:
            if self._state in {WorkerState.UNHEALTHY, WorkerState.STOPPED}:
                self._reject("worker_unavailable", 503)
            if self._state == WorkerState.DRAINING and not allow_draining:
                self._reject("worker_draining", 503)

            queue_limit = (
                self._config.predict_queue_limit
                if kind == WorkKind.PREDICTION
                else self._config.train_queue_limit
            )
            user_limit = (
                self._config.per_user_predict_limit
                if kind == WorkKind.PREDICTION
                else self._config.per_user_train_limit
            )
            if self._queue_depth[kind] >= queue_limit:
                self._reject(f"{kind.value}_queue_full", 503)
            outstanding = self._active_by_user[(kind, user_id)] + sum(
                len(queue)
                for queued_user, queue in self._queues[kind].items()
                if queued_user == user_id
            )
            if outstanding >= user_limit:
                self._reject(f"per_user_{kind.value}_limit", 429)

            user_queue = self._queues[kind].setdefault(user_id, deque())
            user_queue.append(item)
            self._queue_depth[kind] += 1
            self._counters[f"admitted_{kind.value}"] += 1
            self._dispatch_locked(loop)
        try:
            return await asyncio.wait_for(
                asyncio.shield(future), timeout=self._config.request_timeout_seconds
            )
        except asyncio.CancelledError:
            future.cancel()
            self._counters[f"cancelled_{kind.value}"] += 1
            raise
        except asyncio.TimeoutError as exc:
            future.cancel()
            self._counters[f"timed_out_{kind.value}"] += 1
            raise AdmissionRejected(
                f"{kind.value.title()} request timed out", 503, "request_timeout"
            ) from exc

    def _reject(self, reason: str, status_code: int) -> None:
        self._counters[f"rejected_{reason}"] += 1
        raise AdmissionRejected(
            f"Interactive worker admission rejected: {reason}", status_code, reason
        )

    def _kind_available(self, kind: WorkKind) -> bool:
        slot_limit = (
            self._config.predict_slots_per_worker
            if kind == WorkKind.PREDICTION
            else self._config.train_slots_per_worker
        )
        return self._queue_depth[kind] > 0 and self._active[kind] < slot_limit

    def _choose_kind(self) -> Optional[WorkKind]:
        if sum(self._active.values()) >= self._max_active:
            return None
        predict = self._kind_available(WorkKind.PREDICTION)
        train = self._kind_available(WorkKind.TRAINING)
        if predict and train:
            if self._prediction_burst >= self._config.prediction_burst_before_training:
                return WorkKind.TRAINING
            return WorkKind.PREDICTION
        if predict:
            return WorkKind.PREDICTION
        if train:
            return WorkKind.TRAINING
        return None

    def _pop_fair(self, kind: WorkKind) -> Optional[WorkItem[Any]]:
        queues = self._queues[kind]
        attempts = len(queues)
        while attempts and queues:
            user_id, queue = queues.popitem(last=False)
            attempts -= 1
            if not queue:
                continue
            if kind == WorkKind.PREDICTION and self._active_by_user[(kind, user_id)] >= self._config.per_user_predict_limit:
                queues[user_id] = queue
                continue
            item = queue.popleft()
            if queue:
                queues[user_id] = queue
            self._queue_depth[kind] -= 1
            if item.future.cancelled():
                continue
            return item
        return None

    def _dispatch_locked(self, loop: asyncio.AbstractEventLoop) -> None:
        while True:
            kind = self._choose_kind()
            if kind is None:
                return
            item = self._pop_fair(kind)
            if item is None:
                return
            self._active[kind] += 1
            self._active_by_user[(kind, item.user_id)] += 1
            if kind == WorkKind.PREDICTION:
                self._prediction_burst += 1
            else:
                self._prediction_burst = 0
            wait = time.monotonic() - item.enqueued_at
            self._queue_wait_seconds[kind].append(wait)

            started = time.monotonic()
            executor_future = loop.run_in_executor(self._executor, item.fn)

            def completed(
                done: asyncio.Future[Any],
                current: WorkItem[Any] = item,
                started_at: float = started,
            ) -> None:
                duration = time.monotonic() - started_at
                asyncio.create_task(self._finish(current, done, duration))

            executor_future.add_done_callback(completed)

    async def _finish(
        self, item: WorkItem[Any], executor_future: asyncio.Future[Any], duration: float
    ) -> None:
        async with self._lock:
            self._active[item.kind] -= 1
            self._active_by_user[(item.kind, item.user_id)] -= 1
            self._compute_seconds[item.kind].append(duration)
            self._last_seen = time.time()
            self._counters[f"completed_{item.kind.value}"] += 1
            if not item.future.cancelled():
                if executor_future.cancelled():
                    item.future.cancel()
                else:
                    exc = executor_future.exception()
                    if exc is None:
                        item.future.set_result(executor_future.result())
                    else:
                        item.future.set_exception(exc)
            self._dispatch_locked(asyncio.get_running_loop())

    async def cancel_classifier(self, classifier_id: str) -> int:
        cancelled = 0
        async with self._lock:
            for kind in WorkKind:
                for user_id, queue in list(self._queues[kind].items()):
                    kept: Deque[WorkItem[Any]] = deque()
                    while queue:
                        item = queue.popleft()
                        if item.classifier_id == classifier_id:
                            cancelled += 1
                            self._queue_depth[kind] -= 1
                            item.future.cancel()
                        else:
                            kept.append(item)
                    if kept:
                        self._queues[kind][user_id] = kept
                    else:
                        self._queues[kind].pop(user_id, None)
        return cancelled

    async def set_draining(self) -> None:
        async with self._lock:
            if self._state == WorkerState.HEALTHY:
                self._state = WorkerState.DRAINING

    async def set_unhealthy(self, reason: str) -> None:
        async with self._lock:
            self._state = WorkerState.UNHEALTHY
            self._counters["worker_failures"] += 1
            for kind in WorkKind:
                for queue in self._queues[kind].values():
                    while queue:
                        item = queue.popleft()
                        self._queue_depth[kind] -= 1
                        if not item.future.done():
                            item.future.set_exception(
                                AdmissionRejected(
                                    f"Worker failed: {reason}", 503, "worker_failed"
                                )
                            )
                self._queues[kind].clear()

    async def close(self, timeout: float) -> None:
        await self.set_draining()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            async with self._lock:
                if sum(self._active.values()) == 0 and sum(self._queue_depth.values()) == 0:
                    break
            await asyncio.sleep(0.02)
        await self.set_unhealthy("router shutdown")
        self._state = WorkerState.STOPPED
        self._executor.shutdown(wait=False, cancel_futures=True)

    def health(self) -> dict[str, Any]:
        def stats(values: list[float]) -> dict[str, float]:
            if not values:
                return {"count": 0, "average_seconds": 0.0, "max_seconds": 0.0}
            recent = values[-1024:]
            return {
                "count": len(values),
                "average_seconds": sum(recent) / len(recent),
                "max_seconds": max(recent),
            }

        return {
            "worker_id": self.worker_id,
            "state": self._state.value,
            "last_seen_at": self._last_seen,
            "queue_depth": {kind.value: self._queue_depth[kind] for kind in WorkKind},
            "active": {kind.value: self._active[kind] for kind in WorkKind},
            "limits": {
                "prediction_slots": self._config.predict_slots_per_worker,
                "training_slots": self._config.train_slots_per_worker,
                "prediction_queue": self._config.predict_queue_limit,
                "training_queue": self._config.train_queue_limit,
            },
            "counters": dict(self._counters),
            "queue_wait": {kind.value: stats(self._queue_wait_seconds[kind]) for kind in WorkKind},
            "compute_duration": {kind.value: stats(self._compute_seconds[kind]) for kind in WorkKind},
        }


class InteractiveWorker:
    def __init__(self, worker_id: str, weight: float, config: RouterConfig):
        self.worker_id = worker_id
        self.weight = weight
        self.scheduler = FairWorkerScheduler(worker_id, config)
        self._models: Dict[str, Any] = {}
        self._models_lock = threading.RLock()
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._cache_bytes = 0
        self._cache_limit = config.prediction_cache_bytes
        self._cache_counters: Dict[str, int] = defaultdict(int)

    def install_model(self, classifier_id: str, model: Any) -> None:
        with self._models_lock:
            self._models[classifier_id] = model

    def remove_model(self, classifier_id: str) -> None:
        with self._models_lock:
            self._models.pop(classifier_id, None)
            prefix = f"{classifier_id}:"
            for key in [key for key in self._cache if key.startswith(prefix)]:
                self._cache_bytes -= len(self._cache.pop(key))
                self._cache_counters["evictions"] += 1

    def has_model(self, classifier_id: str) -> bool:
        with self._models_lock:
            return classifier_id in self._models

    async def run_with_model(
        self,
        kind: WorkKind,
        user_id: str,
        classifier_id: str,
        fn: Callable[[Any], T],
        cache_key: Optional[str] = None,
    ) -> T:
        if cache_key and kind == WorkKind.PREDICTION:
            with self._models_lock:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    self._cache.move_to_end(cache_key)
                    self._cache_counters["hits"] += 1
                    return cached  # type: ignore[return-value]
                self._cache_counters["misses"] += 1

        def invoke() -> T:
            with self._models_lock:
                model = self._models.get(classifier_id)
            if model is None:
                raise ClassifierGone("Classifier model is unavailable; retraining is required")
            return fn(model)

        result = await self.scheduler.submit(
            kind,
            user_id,
            invoke,
            classifier_id=classifier_id,
            allow_draining=True,
        )
        if cache_key and kind == WorkKind.PREDICTION and isinstance(result, bytes):
            with self._models_lock:
                if self._cache_limit > 0 and len(result) <= self._cache_limit:
                    previous = self._cache.pop(cache_key, None)
                    if previous is not None:
                        self._cache_bytes -= len(previous)
                    self._cache[cache_key] = result
                    self._cache_bytes += len(result)
                    while self._cache_bytes > self._cache_limit and self._cache:
                        _, evicted = self._cache.popitem(last=False)
                        self._cache_bytes -= len(evicted)
                        self._cache_counters["evictions"] += 1
        return result

    def health(self) -> dict[str, Any]:
        result = self.scheduler.health()
        with self._models_lock:
            result.update(
                {
                    "weight": self.weight,
                    "models": len(self._models),
                    "prediction_cache_bytes": self._cache_bytes,
                    "prediction_cache_limit_bytes": self._cache_limit,
                    "prediction_cache": dict(self._cache_counters),
                }
            )
        return result


@dataclass
class TrainingResult:
    classifier_id: str
    generation: int
    worker_id: str
    model: Any


@dataclass
class _IdempotencyEntry:
    fingerprint: str
    task: asyncio.Task[TrainingResult]
    expires_at: float


class SessionRouter:
    """Classifier lifecycle coordinator and strict-affinity worker pool."""

    def __init__(self, config: Optional[RouterConfig] = None, registry: Optional[Registry] = None):
        self.config = config or RouterConfig.from_env()
        self.registry: Registry = registry or InMemoryRegistry(self.config.model_ttl_seconds)
        weights = self.config.worker_weights or (1.0,) * self.config.worker_count
        self.workers: Dict[str, InteractiveWorker] = {
            f"worker-{index + 1}": InteractiveWorker(
                f"worker-{index + 1}", weights[index], self.config
            )
            for index in range(self.config.worker_count)
        }
        self._cleanup_task: Optional[asyncio.Task[None]] = None
        self._idempotency: Dict[tuple[str, str], _IdempotencyEntry] = {}
        self._idempotency_lock = asyncio.Lock()
        self._singleflight: Dict[str, asyncio.Task[Any]] = {}
        self._singleflight_lock = asyncio.Lock()
        self._admission_lock = asyncio.Lock()
        self._outstanding: Dict[tuple[WorkKind, str], int] = defaultdict(int)
        self._global_outstanding: Dict[WorkKind, int] = defaultdict(int)
        self._counters: Dict[str, int] = defaultdict(int)
        self._started_at = time.time()

    def _process_rss_bytes(self) -> Optional[int]:
        try:
            import psutil  # type: ignore

            return int(psutil.Process().memory_info().rss)
        except (ImportError, OSError):
            pass
        try:
            statm = pathlib.Path("/proc/self/statm")
            if statm.exists():
                pages = int(statm.read_text(encoding="ascii").split()[1])
                return pages * int(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, AttributeError):
            pass
        if os.name == "nt":
            try:
                import ctypes
                from ctypes import wintypes

                class ProcessMemoryCounters(ctypes.Structure):
                    _fields_ = [
                        ("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t),
                    ]

                counters = ProcessMemoryCounters()
                counters.cb = ctypes.sizeof(counters)
                handle = ctypes.windll.kernel32.GetCurrentProcess()
                if ctypes.windll.psapi.GetProcessMemoryInfo(
                    handle, ctypes.byref(counters), counters.cb
                ):
                    return int(counters.WorkingSetSize)
            except (AttributeError, OSError):
                pass
        return None

    def _memory_saturated(self) -> bool:
        limit = self.config.worker_memory_high_watermark_bytes
        rss = self._process_rss_bytes()
        return bool(limit and rss is not None and rss >= limit)

    def _select_worker(self) -> InteractiveWorker:
        if self._memory_saturated():
            self._counters["rejected_memory_high_watermark"] += 1
            raise AdmissionRejected(
                "Worker memory high-watermark reached", 503, "memory_high_watermark"
            )
        candidates = [
            worker
            for worker in self.workers.values()
            if worker.scheduler.can_accept_training()
        ]
        if not candidates:
            self._counters["rejected_no_worker"] += 1
            raise AdmissionRejected("No healthy worker has training capacity", 503, "no_worker")
        return min(candidates, key=lambda worker: (worker.scheduler.load_score(worker.weight), worker.worker_id))

    async def start(self) -> None:
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.config.cleanup_interval_seconds)
                await self.expire_once()
                await self._purge_idempotency()
        except asyncio.CancelledError:
            return

    async def close(self) -> None:
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            await asyncio.gather(self._cleanup_task, return_exceptions=True)
        await asyncio.gather(
            *(
                worker.scheduler.close(self.config.shutdown_timeout_seconds)
                for worker in self.workers.values()
            )
        )

    async def _purge_idempotency(self) -> None:
        now = time.time()
        async with self._idempotency_lock:
            for key, entry in list(self._idempotency.items()):
                if entry.expires_at <= now and entry.task.done():
                    self._idempotency.pop(key, None)

    @staticmethod
    def request_fingerprint(value: Any) -> str:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    async def train(
        self,
        user_id: str,
        feature_config: Mapping[str, Any],
        train_fn: Callable[[], Any],
        *,
        idempotency_key: Optional[str] = None,
        fingerprint: Optional[str] = None,
    ) -> TrainingResult:
        if not idempotency_key:
            return await self._train_once(user_id, feature_config, train_fn)
        key = (user_id, idempotency_key)
        request_hash = fingerprint or self.request_fingerprint(feature_config)
        async with self._idempotency_lock:
            existing = self._idempotency.get(key)
            if existing is not None:
                if existing.fingerprint != request_hash:
                    raise ClassifierConflict(
                        "Idempotency-Key was already used for a different training request"
                    )
                task = existing.task
                self._counters["idempotency_replays"] += 1
            else:
                task = asyncio.create_task(self._train_once(user_id, feature_config, train_fn))
                self._idempotency[key] = _IdempotencyEntry(
                    request_hash,
                    task,
                    time.time() + self.config.idempotency_ttl_seconds,
                )
        return await asyncio.shield(task)

    async def _train_once(
        self, user_id: str, feature_config: Mapping[str, Any], train_fn: Callable[[], Any]
    ) -> TrainingResult:
        await self._acquire_admission(WorkKind.TRAINING, user_id)
        record: Optional[ClassifierRecord] = None
        try:
            worker = self._select_worker()
            record = self.registry.allocate(user_id, worker.worker_id, feature_config)
            model = await worker.scheduler.submit(
                WorkKind.TRAINING,
                user_id,
                train_fn,
                classifier_id=record.classifier_id,
                allow_draining=True,
            )
            worker.install_model(record.classifier_id, model)
            try:
                previous = self.registry.publish_ready(record.classifier_id)
            except Exception:
                worker.remove_model(record.classifier_id)
                raise
            if previous is not None:
                old_worker = self.workers.get(previous.worker_id)
                if old_worker is not None:
                    await old_worker.scheduler.cancel_classifier(previous.classifier_id)
                    old_worker.remove_model(previous.classifier_id)
            self._counters["training_succeeded"] += 1
            return TrainingResult(
                record.classifier_id, record.generation, record.worker_id, model
            )
        except asyncio.CancelledError:
            if record is not None:
                self.registry.mark_failed(record.classifier_id, "Training request was cancelled")
                await worker.scheduler.cancel_classifier(record.classifier_id)
                worker.remove_model(record.classifier_id)
            self._counters["training_cancelled"] += 1
            raise
        except (AdmissionRejected, ClassifierConflict):
            if record is not None:
                self.registry.mark_failed(record.classifier_id, "Training was not admitted or was superseded")
            self._counters["training_rejected_or_stale"] += 1
            raise
        except Exception as exc:
            if record is not None:
                self.registry.mark_failed(record.classifier_id, str(exc))
                worker.remove_model(record.classifier_id)
            self._counters["training_failed"] += 1
            raise
        finally:
            await self._release_admission(WorkKind.TRAINING, user_id)

    async def _acquire_admission(self, kind: WorkKind, user_id: str) -> None:
        user_limit = (
            self.config.per_user_predict_limit
            if kind == WorkKind.PREDICTION
            else self.config.per_user_train_limit
        )
        slots = (
            self.config.predict_slots_per_worker
            if kind == WorkKind.PREDICTION
            else self.config.train_slots_per_worker
        ) * self.config.worker_count
        queue_limit = (
            self.config.predict_queue_limit
            if kind == WorkKind.PREDICTION
            else self.config.train_queue_limit
        )
        async with self._admission_lock:
            if self._outstanding[(kind, user_id)] >= user_limit:
                self._counters[f"rejected_per_user_{kind.value}"] += 1
                raise AdmissionRejected(
                    f"Per-user {kind.value} limit reached",
                    429,
                    f"per_user_{kind.value}_limit",
                )
            if self._global_outstanding[kind] >= slots + queue_limit:
                self._counters[f"rejected_global_{kind.value}"] += 1
                raise AdmissionRejected(
                    f"Global {kind.value} queue is full",
                    503,
                    f"global_{kind.value}_queue_full",
                )
            self._outstanding[(kind, user_id)] += 1
            self._global_outstanding[kind] += 1

    async def _release_admission(self, kind: WorkKind, user_id: str) -> None:
        async with self._admission_lock:
            self._outstanding[(kind, user_id)] = max(
                0, self._outstanding[(kind, user_id)] - 1
            )
            self._global_outstanding[kind] = max(
                0, self._global_outstanding[kind] - 1
            )

    def authorize(
        self, classifier_id: str, user_id: str, generation: Optional[int] = None
    ) -> ClassifierRecord:
        return self.registry.resolve(classifier_id, user_id, generation)

    async def run_model_task(
        self,
        classifier_id: str,
        user_id: str,
        kind: WorkKind,
        fn: Callable[[Any], T],
        *,
        generation: Optional[int] = None,
        cache_key: Optional[str] = None,
        singleflight_key: Optional[str] = None,
    ) -> T:
        record = self.registry.resolve(classifier_id, user_id, generation)
        worker = self.workers.get(record.worker_id)
        if worker is None or worker.scheduler.state in {WorkerState.UNHEALTHY, WorkerState.STOPPED}:
            await self.mark_worker_unhealthy(record.worker_id, "assigned worker unavailable")
            raise ClassifierGone("Classifier worker was lost; retraining is required")

        async def execute() -> T:
            await self._acquire_admission(kind, user_id)
            try:
                return await worker.run_with_model(
                    kind, user_id, classifier_id, fn, cache_key=cache_key
                )
            finally:
                await self._release_admission(kind, user_id)

        if not singleflight_key:
            return await execute()
        async with self._singleflight_lock:
            task = self._singleflight.get(singleflight_key)
            if task is None:
                task = asyncio.create_task(execute())
                self._singleflight[singleflight_key] = task
                self._counters["singleflight_started"] += 1
            else:
                self._counters["singleflight_joined"] += 1
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                async with self._singleflight_lock:
                    if self._singleflight.get(singleflight_key) is task:
                        self._singleflight.pop(singleflight_key, None)

    async def run_utility(
        self, kind: WorkKind, user_id: str, fn: Callable[[], T]
    ) -> T:
        if self._memory_saturated():
            raise AdmissionRejected(
                "Worker memory high-watermark reached", 503, "memory_high_watermark"
            )
        candidates = [
            worker
            for worker in self.workers.values()
            if worker.scheduler.state == WorkerState.HEALTHY
        ]
        if not candidates:
            raise AdmissionRejected("No healthy interactive worker", 503, "no_worker")
        worker = min(
            candidates,
            key=lambda candidate: (
                candidate.scheduler.load_score(candidate.weight),
                candidate.worker_id,
            ),
        )
        await self._acquire_admission(kind, user_id)
        try:
            return await worker.scheduler.submit(kind, user_id, fn)
        finally:
            await self._release_admission(kind, user_id)

    async def expire_once(self, now: Optional[float] = None) -> list[ClassifierRecord]:
        expired = self.registry.expire_due(now)
        for record in expired:
            worker = self.workers.get(record.worker_id)
            if worker is not None:
                await worker.scheduler.cancel_classifier(record.classifier_id)
                worker.remove_model(record.classifier_id)
        return expired

    async def drain_worker(self, worker_id: str) -> None:
        worker = self.workers.get(worker_id)
        if worker is None:
            raise ClassifierUnknown(f"Worker {worker_id} not found")
        await worker.scheduler.set_draining()
        self.registry.mark_worker_draining(worker_id)

    async def mark_worker_unhealthy(self, worker_id: str, reason: str) -> None:
        worker = self.workers.get(worker_id)
        if worker is not None:
            await worker.scheduler.set_unhealthy(reason)
        lost = self.registry.mark_worker_lost(worker_id, reason)
        if worker is not None:
            for record in lost:
                worker.remove_model(record.classifier_id)
        self._counters["worker_failures"] += 1
        self._counters["classifiers_lost"] += len(lost)

    def health(self) -> dict[str, Any]:
        workers = [worker.health() for worker in self.workers.values()]
        states: Dict[str, int] = defaultdict(int)
        for worker in workers:
            states[worker["state"]] += 1
        rss = self._process_rss_bytes()
        saturated = self._memory_saturated()
        return {
            "status": "ok" if states[WorkerState.HEALTHY.value] else "degraded",
            "scope": "single_process_single_vm",
            "started_at": self._started_at,
            "registry": self.registry.snapshot(),
            "worker_summary": dict(states),
            "workers": workers,
            "memory": {
                "process_rss_bytes": rss,
                "high_watermark_bytes": self.config.worker_memory_high_watermark_bytes,
                "saturated": saturated,
            },
            "counters": dict(self._counters),
            "admission": {
                "global_outstanding": {
                    kind.value: self._global_outstanding[kind] for kind in WorkKind
                }
            },
        }
