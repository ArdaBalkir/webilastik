# Interactive session router

## Scope and safety boundary

The compute server has one authoritative session router for interactive
training, tile prediction, and the legacy local export endpoints. It owns a
fixed set of logical workers. Each logical worker has a bounded fair scheduler
and a model table. A classifier record stores its assigned worker, and every
operation using that classifier returns to that worker.

This implementation is safe for **one FastAPI process on one VM**. Workers are
thread-backed logical workers in that process; they are not independently
addressable operating-system processes. The locked in-memory registry is the
authority within that process only. Router restart loses unsnapshotted models.
Run `uvicorn` with one process/worker.

Do not put multiple router processes behind a load balancer with this registry.
Multi-process or multi-VM operation still requires:

- a shared registry (for example PostgreSQL or Redis) with atomic generation,
  assignment, publish, expiry, and worker-loss transactions;
- shared worker discovery, leases, heartbeats, and circuit-breaker state;
- remotely addressable workers and authenticated router-to-worker requests;
- durable model snapshots plus restore/migration, or stable routing to the
  original model-owning worker;
- distributed queue admission and tenant limits; and
- a decision about cache and single-flight coordination across routers.

The `Registry` protocol in `backend/session_router.py` identifies the metadata
operations that a shared implementation must provide. Replacing only the
registry is not sufficient for horizontal operation because workers and queues
are still local.

## Request flow

Training allocates a monotonically increasing per-user generation and selects
the least-loaded healthy, non-draining worker (adjusted by configured worker
weight). The existing ready classifier remains available during retraining.
The new model is installed and published only when training succeeds and its
generation is still the newest request. A late older result is discarded with
HTTP 409. Successful publication removes the superseded model and queued work.

Prediction resolves ownership and lifecycle state in the registry, then runs
on the assigned worker. Identical in-flight tile requests share one operation;
completed PNGs use a byte-bounded per-worker LRU. Optional `generation` query
parameters let clients reject stale view/model work.

Each worker has separate prediction and training queues sharing an explicitly
sized executor. Scheduling is round-robin between users. Predictions win when
both classes are runnable, but a configurable prediction burst limit guarantees
training progress. Admission applies aggregate pool limits and per-user limits
before work enters a worker queue. Rejection is immediate and includes
`Retry-After: 1`.

HTTP behavior:

| Status | Meaning |
| --- | --- |
| 403 | classifier or export job belongs to another user |
| 404 | identifier was never known |
| 409 | classifier is training, generation is stale, or idempotency key conflicts |
| 410 | classifier expired, was replaced, failed, or its worker/model was lost |
| 429 | per-user admission limit reached |
| 503 | global queue, worker availability, memory, or timeout backpressure |

`POST /train` and `POST /train-multi` accept `Idempotency-Key`. A key is scoped
to the authenticated user. Reusing it with the same request joins/replays the
same training task; using it with a different request returns 409. Existing
request bodies and the existing `classifier_id` and `num_classes` response
fields are unchanged. `generation` and `worker_id` are additive response fields.

## Lifecycle and failure behavior

Classifier states are `allocating`, `training`, `ready`, `draining`, `expired`,
`failed`, `lost`, and `deleted`. Active TTL is refreshed by model use. Cleanup
removes expired worker-side models and active registry records while retaining a
bounded-time tombstone so callers receive deterministic 410 responses.

Draining workers receive no new classifier assignments. Their existing
classifiers enter `draining` and remain usable until expiry or shutdown. Marking
a worker unhealthy cancels its queued work, removes its models, and marks all
assigned classifiers `lost`; subsequent use returns 410 with a retraining
message. Ordinary request failures do not declare an entire worker dead.

On server shutdown workers drain for `WORKER_SHUTDOWN_TIMEOUT_SECONDS`, after
which queued work is failed and executors stop accepting work.

## Configuration

All defaults are fixed budgets; none are derived from `os.cpu_count()`.

| Variable | Default | Purpose |
| --- | ---: | --- |
| `INTERACTIVE_WORKER_COUNT` | `2` | logical model-owning workers |
| `INTERACTIVE_WORKER_WEIGHTS` | equal | comma-separated positive assignment weights |
| `PREDICT_SLOTS_PER_WORKER` | `2` | concurrent prediction slots per worker |
| `TRAIN_SLOTS_PER_WORKER` | `1` | concurrent training slots per worker |
| `PREDICT_QUEUE_LIMIT` | `32` | aggregate queued prediction budget (also caps each worker) |
| `TRAIN_QUEUE_LIMIT` | `8` | aggregate queued training budget (also caps each worker) |
| `PER_USER_PREDICT_LIMIT` | `8` | pool-wide outstanding predictions per user |
| `PER_USER_TRAIN_LIMIT` | `2` | pool-wide outstanding trainings per user |
| `MODEL_TTL_SECONDS` | `1800` | idle model lifetime |
| `CLEANUP_INTERVAL_SECONDS` | `30` | registry/cache cleanup cadence |
| `REQUEST_TIMEOUT_SECONDS` | `300` | maximum queue plus compute wait |
| `WORKER_SHUTDOWN_TIMEOUT_SECONDS` | `20` | graceful drain deadline |
| `WORKER_MEMORY_HIGH_WATERMARK_MB` | `0` | stop admission at process RSS; `0` disables |
| `COMPUTE_THREADS_PER_TASK` | `1` | sklearn and local tile-export inner threads |
| `PREDICTION_CACHE_MB` | `32` | per-worker PNG cache byte budget; `0` disables |
| `PREDICTION_BURST_BEFORE_TRAINING` | `4` | prediction priority burst before queued training |
| `IDEMPOTENCY_TTL_SECONDS` | `600` | completed/in-flight training key retention |

`COMPUTE_THREADS_PER_TASK=1` is the recommended interactive setting. Increase
it only after reducing outer worker/slot concurrency to keep the total CPU
budget within guaranteed VM resources.

## Health and telemetry

`GET /health` is always handled by FastAPI and does not enter a compute queue.
It reports registry state counts, healthy/draining/unhealthy workers, per-class
queue depth and active counts, admission/completion/failure counters,
queue-wait and compute-duration summaries, model/cache counts, cache
hits/misses/evictions, process RSS, memory saturation, and the explicit
`single_process_single_vm` scope.

Logs include classifier and worker identifiers. User IDs are useful for access
control and debugging but are not used as metric label keys. These local
counters intentionally avoid a mandatory external telemetry service.

## Bulk export

Browser-driven bulk export remains on the existing session allocator:

`frontend -> backend.session_allocator -> SSH sbatch -> SLURM -> headless_cli`

The allocator `/health` identifies itself as `slurm_bulk_export_allocator`.
The compute server's `/batch-export`, `/export`, `/export-zip`, and
`/headless-run` endpoints remain for compatibility/local testing and now obey
classifier affinity, but they are not the production bulk path.
