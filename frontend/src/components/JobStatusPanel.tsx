import { h } from "preact";
import { useEffect } from "preact/hooks";
import { useSignal } from "@preact/signals";
import * as state from "../state";
import { SessionAllocatorClient } from "../api";

const POLL_MS = 5_000;
const LOG_REFRESH_MS = 10_000;
const TERMINAL = new Set(["done", "error", "cancelled"]);

function statusLabel(s: string): string {
  switch (s) {
    case "pending":   return "⏳ Pending";
    case "running":   return "⚡ Running";
    case "done":      return "✅ Done";
    case "error":     return "❌ Error";
    case "cancelled": return "🚫 Cancelled";
    default:          return s;
  }
}

function makeClient() {
  return new SessionAllocatorClient(
    state.allocatorUrl.value,
    state.bearerToken.value,
  );
}

export function JobStatusPanel() {
  const job = state.activeHpcJob.value;
  const history = state.hpcJobHistory.value;
  const showLog = useSignal(false);
  const log = useSignal("");
  const cancelling = useSignal(false);

  const isTerminal = !job || TERMINAL.has(job.status);

  // Poll for status updates while job is active
  useEffect(() => {
    if (!job || TERMINAL.has(job.status)) return;

    let stopped = false;

    (async () => {
      while (!stopped) {
        await new Promise((r) => setTimeout(r, POLL_MS));
        if (stopped) break;
        try {
          const updated = await makeClient().getHeadlessJob(job.job_id);
          state.updateHpcJob(job.job_id, updated);
          // Update active reference so UI re-renders
          state.activeHpcJob.value = { ...state.activeHpcJob.value!, ...updated };
          if (TERMINAL.has(updated.status)) break;
        } catch {
          // network blip — keep trying
        }
      }
    })();

    return () => { stopped = true; };
  }, [job?.job_id]);

  // Fetch + periodically refresh log when panel is expanded
  useEffect(() => {
    if (!showLog.value || !job) return;

    let stopped = false;

    async function fetchLog() {
      if (stopped) return;
      try {
        const text = await makeClient().getJobLog(job!.job_id, 40);
        if (!stopped) log.value = text;
      } catch {
        if (!stopped) log.value = "(could not fetch log)";
      }
    }

    fetchLog();
    if (isTerminal) return;
    const id = setInterval(fetchLog, LOG_REFRESH_MS);
    return () => { stopped = true; clearInterval(id); };
  }, [showLog.value, job?.job_id, isTerminal]);

  async function handleCancel() {
    if (!job) return;
    cancelling.value = true;
    try {
      await makeClient().cancelHeadlessJob(job.job_id);
      state.updateHpcJob(job.job_id, { status: "cancelled" });
      state.activeHpcJob.value = { ...job, status: "cancelled" };
    } catch (e) {
      alert(`Cancel failed: ${e}`);
    }
    cancelling.value = false;
  }

  if (!job && history.length === 0) return null;

  return (
    <section class="panel job-panel">
      <h2>HPC Jobs</h2>

      {job && (
        <div class={`job-card job-${job.status}`}>
          <div class="job-header">
            <span class="job-badge">{statusLabel(job.status)}</span>
            <span class="job-slurm" title={job.job_id}>
              #{job.slurm_job_id}
            </span>
            {!isTerminal && (
              <button
                class="btn-sm danger"
                onClick={handleCancel}
                disabled={cancelling.value}
              >
                {cancelling.value ? "…" : "Cancel"}
              </button>
            )}
          </div>

          <button
            class="btn-sm"
            onClick={() => { showLog.value = !showLog.value; }}
          >
            {showLog.value ? "Hide log ▲" : "Show log ▼"}
          </button>

          {showLog.value && (
            <pre class="job-log">{log.value || "(log not yet available)"}</pre>
          )}
        </div>
      )}

      {history.length > 0 && (
        <details class="job-history">
          <summary>History ({history.length})</summary>
          {history.map((r) => (
            <div key={r.job_id} class={`job-hist-row job-${r.status}`}>
              <span class="job-badge job-badge-sm">{statusLabel(r.status)}</span>
              <span class="job-slurm">#{r.slurm_job_id}</span>
              <span class="job-time">
                {new Date(r.created_at).toLocaleTimeString()}
              </span>
            </div>
          ))}
        </details>
      )}
    </section>
  );
}
