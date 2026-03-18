/**
 * ExportPanel — HPC batch export via session allocator
 *
 * Flow:
 *   1. User fills p_source + output_dir (pre-seeded from ?p_source= / ?output_dir=)
 *   2. "Run on HPC" → POST to allocator /headless-jobs (includes annotations)
 *   3. Poll allocator every 5s for SLURM status
 *   4. "Show log" tail-fetches the SLURM job log over SSH via the allocator
 *
 * The compute server (server.py) is NOT used for export — everything runs on HPC.
 */
import { h } from "preact";
import { useEffect } from "preact/hooks";
import { useSignal } from "@preact/signals";
import { SessionAllocatorClient, featureConfigToFilters } from "../api";
import type { HpcJobStatus } from "../api";
import * as state from "../state";

const STATUS_ICON: Record<string, string> = {
  pending: "⏳",
  running: "🔄",
  done: "✅",
  error: "❌",
  cancelled: "🚫",
};

export function ExportPanel() {
  const open = useSignal(false);
  const pSource = useSignal(state.pSourceUrl.value);
  const outputDir = useSignal(state.outputDirUrl.value);
  const allocatorUrlInput = useSignal(state.allocatorUrl.value);

  const jobStatus = useSignal<HpcJobStatus | null>(null);
  const msg = useSignal("");
  const log = useSignal("");
  const showLog = useSignal(false);
  const polling = useSignal<ReturnType<typeof setInterval> | null>(null);

  function makeClient() {
    return new SessionAllocatorClient(
      allocatorUrlInput.value.trim(),
      state.bearerToken.value,
    );
  }

  function stopPolling() {
    if (polling.value) {
      clearInterval(polling.value);
      polling.value = null;
    }
  }

  function startPolling(jobId: string) {
    stopPolling();
    polling.value = setInterval(async () => {
      try {
        const s = await makeClient().getHeadlessJob(jobId);
        jobStatus.value = s;
        if (s.status === "done") {
          stopPolling();
          msg.value = `✅ Done — SLURM job ${s.slurm_job_id} completed.`;
        } else if (s.status === "error" || s.status === "cancelled") {
          stopPolling();
          msg.value = `❌ Job ${s.slurm_job_id} ended with status: ${s.slurm_state}`;
        }
      } catch (e) {
        msg.value = `Poll error: ${e}`;
      }
    }, 5000);
  }

  // Sync allocator URL signal when user edits the input
  useEffect(() => {
    const unsub = allocatorUrlInput.subscribe((v) => {
      state.allocatorUrl.value = v;
    });
    return unsub;
  }, []);

  // Cleanup on unmount
  useEffect(() => () => stopPolling(), []);

  async function submitJob() {
    if (!pSource.value.trim() || !outputDir.value.trim()) {
      msg.value = "Set both source directory and output directory.";
      return;
    }

    // Collect all annotations from all sources
    const bySource = { ...state.strokesBySource.value };
    if (state.dziUrl.value && state.strokes.value.length > 0) {
      bySource[state.dziUrl.value] = state.strokes.value;
    }

    const annotations = Object.entries(bySource)
      .filter(([, ss]) => ss.length > 0)
      .map(([dzip_url, ss]) => ({
        dzip_url,
        strokes: ss.map((s) => ({
          label: s.labelId,
          points: s.points,
        })),
      }));

    if (annotations.length === 0) {
      msg.value = "No annotations — paint some strokes first.";
      return;
    }

    const fc = state.featureConfig.value;
    const features = {
      filters: featureConfigToFilters(fc),
      scales: fc.scales,
    };

    msg.value = `Submitting HPC job (${annotations.length} annotated image(s))…`;
    jobStatus.value = null;
    log.value = "";
    showLog.value = false;

    try {
      const res = await makeClient().submitHeadlessJob({
        annotations,
        features,
        p_source: pSource.value.trim(),
        output_dir: outputDir.value.trim(),
      });
      jobStatus.value = res;
      msg.value = `Submitted → SLURM job ${res.slurm_job_id}`;
      startPolling(res.job_id);
    } catch (e) {
      msg.value = `Submission failed: ${e}`;
    }
  }

  async function cancelJob() {
    const js = jobStatus.value;
    if (!js) return;
    try {
      await makeClient().cancelHeadlessJob(js.job_id);
      stopPolling();
      msg.value = `Cancelled SLURM job ${js.slurm_job_id}`;
      jobStatus.value = { ...js, status: "cancelled" };
    } catch (e) {
      msg.value = `Cancel failed: ${e}`;
    }
  }

  async function fetchLog() {
    const js = jobStatus.value;
    if (!js) return;
    try {
      log.value = "Loading…";
      showLog.value = true;
      log.value = await makeClient().getJobLog(js.job_id, 100);
    } catch (e) {
      log.value = `Could not fetch log: ${e}`;
    }
  }

  const js = jobStatus.value;
  const isActive = js && (js.status === "pending" || js.status === "running");

  return (
    <section class="panel collapsible">
      <button
        class="collapsible-header"
        onClick={() => (open.value = !open.value)}
      >
        <span>HPC Export</span>
        {js && (
          <span style={{ marginLeft: 6 }}>{STATUS_ICON[js.status] ?? ""}</span>
        )}
        <span class="chevron">{open.value ? "▲" : "▼"}</span>
      </button>

      {open.value && (
        <div class="collapsible-body">
          <label class="hint">Allocator URL</label>
          <input
            class="input-url"
            value={allocatorUrlInput.value}
            placeholder="http://localhost:8001"
            onInput={(e: Event) =>
              (allocatorUrlInput.value = (e.target as HTMLInputElement).value)
            }
          />

          <label class="hint">Source directory (p_source)</label>
          <input
            class="input-url"
            value={pSource.value}
            placeholder="https://data-proxy.ebrains.eu/api/v1/buckets/…/images/"
            onInput={(e: Event) =>
              (pSource.value = (e.target as HTMLInputElement).value)
            }
          />

          <label class="hint">Output directory</label>
          <input
            class="input-url"
            value={outputDir.value}
            placeholder="https://data-proxy.ebrains.eu/api/v1/buckets/…/segmentations/"
            onInput={(e: Event) =>
              (outputDir.value = (e.target as HTMLInputElement).value)
            }
          />

          <div class="row" style={{ gap: 6, marginTop: 8 }}>
            <button
              class="btn btn-train"
              onClick={submitJob}
              disabled={!!isActive}
            >
              {isActive ? "Running on HPC…" : "Run on HPC"}
            </button>
            {isActive && (
              <button class="btn-sm danger" onClick={cancelJob}>
                Cancel
              </button>
            )}
          </div>

          {msg.value && (
            <p class={`status ${js?.status === "error" ? "error" : ""}`}>
              {msg.value}
            </p>
          )}

          {js && (
            <div class="hpc-job-info">
              <p class="hint">
                SLURM {js.slurm_job_id} &nbsp;·&nbsp;
                <strong>{js.slurm_state}</strong>
                {js.status === "running" && " 🔄"}
              </p>
              <div class="row" style={{ gap: 6 }}>
                <button class="btn-sm" onClick={fetchLog}>
                  {showLog.value ? "Refresh log" : "Show log"}
                </button>
                {showLog.value && (
                  <button
                    class="btn-sm"
                    onClick={() => (showLog.value = false)}
                  >
                    Hide log
                  </button>
                )}
              </div>
              {showLog.value && log.value && (
                <pre class="job-log">{log.value}</pre>
              )}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
