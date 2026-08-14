import { h } from "preact";
import { useEffect } from "preact/hooks";
import { useSignal } from "@preact/signals";
import * as state from "../state";
import type { HpcJobRecord } from "../types";
import { SessionAllocatorClient } from "../api";

const POLL_MS = 15_000;
const LOG_REFRESH_MS = 15_000;
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

function formatJobDateTime(timestamp: number): string {
  return new Date(timestamp).toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

// ── Single job card ────────────────────────────────────────────────────────────

function JobCard({ job }: { job: HpcJobRecord }) {
  const showLog = useSignal(false);
  const log = useSignal("");
  const cancelling = useSignal(false);
  const isTerminal = TERMINAL.has(job.status);

  // Fetch + refresh log when expanded
  useEffect(() => {
    if (!showLog.value) return;
    let stopped = false;

    async function fetchLog() {
      if (stopped) return;
      try {
        const text = await makeClient().getJobLog(job.job_id, 40);
        if (!stopped) log.value = text;
      } catch {
        if (!stopped) log.value = "(could not fetch log)";
      }
    }

    fetchLog();
    if (isTerminal) return;
    const id = setInterval(fetchLog, LOG_REFRESH_MS);
    return () => { stopped = true; clearInterval(id); };
  }, [showLog.value, job.job_id, isTerminal]);

  async function handleCancel() {
    cancelling.value = true;
    try {
      await makeClient().cancelHeadlessJob(job.job_id);
      state.updateHpcJob(job.job_id, { status: "cancelled" });
    } catch (e) {
      alert(`Cancel failed: ${e}`);
    }
    cancelling.value = false;
  }

  return (
    <div class={`job-card job-${job.status}`}>
      <div class="job-header">
        <span class="job-badge">{statusLabel(job.status)}</span>
        <span class="job-slurm" title={job.job_id}>#{job.slurm_job_id}</span>
        <span class="job-time">{formatJobDateTime(job.created_at)}</span>
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

      <button class="btn-sm" onClick={() => { showLog.value = !showLog.value; }}>
        {showLog.value ? "Hide log ▲" : "Show log ▼"}
      </button>

      {showLog.value && (
        <pre class="job-log">{log.value || "(log not yet available)"}</pre>
      )}
    </div>
  );
}

// ── Panel ──────────────────────────────────────────────────────────────────────

export function JobStatusPanel() {
  const history = state.hpcJobHistory.value;

  // Poll all non-terminal jobs every POLL_MS
  useEffect(() => {
    let stopped = false;

    async function pollAll() {
      const active = state.hpcJobHistory.value.filter(
        (r) => !TERMINAL.has(r.status),
      );
      for (const job of active) {
        if (stopped) break;
        try {
          const updated = await makeClient().getHeadlessJob(job.job_id);
          state.updateHpcJob(job.job_id, updated);
        } catch (e) {
          if (String(e).includes("404")) {
            state.updateHpcJob(job.job_id, {
              status: "error",
              slurm_state: "UNKNOWN",
            });
          }
          // other blip — ignore, retry next round
        }
      }
    }

    // Poll immediately once, then on interval
    pollAll();
    const id = setInterval(() => {
      if (!stopped) pollAll();
    }, POLL_MS);

    return () => { stopped = true; clearInterval(id); };
  }, []); // runs once; reads live signal inside so always sees latest jobs

  if (history.length === 0) return null;

  const active = history.filter((r) => !TERMINAL.has(r.status));
  const done = history.filter((r) => TERMINAL.has(r.status));

  return (
    <section class="panel job-panel">
      <h2>HPC Jobs</h2>

      {/* Active jobs — one card each, all polled */}
      {active.map((job) => (
        <JobCard key={job.job_id} job={job} />
      ))}

      {/* Completed jobs — collapsible history */}
      {done.length > 0 && (
        <details class="job-history">
          <summary>Completed ({done.length})</summary>
          {done.map((r) => (
            <div key={r.job_id} class={`job-hist-row job-${r.status}`}>
              <span class="job-badge job-badge-sm">{statusLabel(r.status)}</span>
              <span class="job-slurm">#{r.slurm_job_id}</span>
              <span class="job-time">
                {formatJobDateTime(r.created_at)}
              </span>
            </div>
          ))}
        </details>
      )}
    </section>
  );
}
