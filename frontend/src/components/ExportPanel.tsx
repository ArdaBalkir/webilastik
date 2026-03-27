/**
 * ExportPanel — HPC batch export via session allocator
 *
 * Features:
 *  - Health indicator (pings allocator on open)
 *  - Job submission with auto-polling
 *  - Live per-image progress parsed from SLURM log
 *  - Job history persisted to localStorage (last 20 jobs)
 *  - Log viewer per job
 */
import { h } from "preact";
import { useEffect } from "preact/hooks";
import { useSignal } from "@preact/signals";
import { SessionAllocatorClient, featureConfigToFilters } from "../api";
import type { HpcJobStatus } from "../api";
import type { HpcJobRecord } from "../types";
import * as state from "../state";

const STATUS_ICON: Record<string, string> = {
  pending: "⏳",
  running: "🔄",
  done: "✅",
  error: "❌",
  cancelled: "🚫",
};

// ── Log parser ──────────────────────────────────────────────────────────────

interface ImageProgress {
  idx: number;      // 1-based image index
  total: number;    // total images
  name: string;     // e.g. "79556738_s306.jpg.dzip"
  tilesDone: number;
  tilesTotal: number;
  rate: string;     // e.g. "82.6/s"
  done: boolean;    // ✓ done seen
  failed: boolean;  // ✗ FAILED seen
}

function parseLogProgress(log: string): ImageProgress[] {
  const images: Map<number, ImageProgress> = new Map();

  for (const line of log.split("\n")) {
    // Match "[N/M] name.dzip" — image header
    const hdr = line.match(/\[(\d+)\/(\d+)\]\s+(\S+)/);
    if (hdr) {
      const idx = parseInt(hdr[1]);
      const total = parseInt(hdr[2]);
      const name = hdr[3];
      if (!images.has(idx)) {
        images.set(idx, { idx, total, name, tilesDone: 0, tilesTotal: 0, rate: "", done: false, failed: false });
      } else {
        const img = images.get(idx)!;
        img.name = name;
        img.total = total;
      }
      continue;
    }

    // Match "tiles X/Y  rate/s" — tile progress
    const tiles = line.match(/tiles\s+(\d+)\/(\d+)\s+([\d.]+\/s)/);
    if (tiles) {
      // Apply to the last (highest-index) image
      const last = [...images.values()].pop();
      if (last) {
        last.tilesDone = parseInt(tiles[1]);
        last.tilesTotal = parseInt(tiles[2]);
        last.rate = tiles[3];
      }
      continue;
    }

    // Match "✓ done" — image completed
    if (line.includes("✓ done") || line.includes("done")) {
      const last = [...images.values()].pop();
      if (last && !last.done) {
        last.done = true;
        last.tilesDone = last.tilesTotal || last.tilesDone;
      }
    }

    // Match "✗ FAILED"
    if (line.includes("✗ FAILED") || line.includes("FAILED")) {
      const last = [...images.values()].pop();
      if (last) last.failed = true;
    }
  }

  // Build ordered array, filling in images we haven't seen yet
  const result = [...images.values()].sort((a, b) => a.idx - b.idx);
  return result;
}

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

  const activeJob = useSignal<HpcJobStatus | null>(null);
  const msg = useSignal("");
  const polling = useSignal<ReturnType<typeof setInterval> | null>(null);

  // Per-job log viewer
  const logJobId = useSignal<string | null>(null);
  const logText = useSignal("");
  const logLoading = useSignal(false);
  const showHistory = useSignal(false);

  // Live progress parsed from log
  const imageProgress = useSignal<ImageProgress[]>([]);
  const showRawLog = useSignal(false);

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
    // Immediately fetch log once
    fetchLogSilent(jobId);
    polling.value = setInterval(async () => {
      try {
        const s = await makeClient().getHeadlessJob(jobId);
        activeJob.value = s;
        state.updateHpcJob(jobId, { status: s.status, slurm_state: s.slurm_state });
        // Always fetch log while active
        await fetchLogSilent(jobId);
        if (s.status === "done") {
          stopPolling();
          msg.value = `✅ Done — SLURM ${s.slurm_job_id} completed.`;
        } else if (s.status === "error" || s.status === "cancelled") {
          stopPolling();
          msg.value = `❌ SLURM ${s.slurm_job_id}: ${s.slurm_state}`;
        }
      } catch (e) {
        msg.value = `Poll error: ${e}`;
      }
    }, 5000);
  }

  /** Fetch log without UI loading state — used by auto-poll. */
  async function fetchLogSilent(jobId: string) {
    try {
      const text = await makeClient().getJobLog(jobId, 200);
      logText.value = text;
      logJobId.value = jobId;
      imageProgress.value = parseLogProgress(text);
    } catch {
      // ignore transient failures during polling
    }
  }

  async function checkHealth() {
    state.hpcAvailable.value = null;
    state.hpcAvailable.value = await makeClient().checkHealth();
  }

  useEffect(() => {
    const unsub = allocatorUrlInput.subscribe((v) => {
      state.allocatorUrl.value = v;
    });
    return unsub;
  }, []);

  useEffect(() => () => stopPolling(), []);

  async function submitJob() {
    if (!pSource.value.trim() || !outputDir.value.trim()) {
      msg.value = "Set both source directory and output directory.";
      return;
    }

    const bySource = { ...state.strokesBySource.value };
    if (state.dziUrl.value && state.strokes.value.length > 0) {
      bySource[state.dziUrl.value] = state.strokes.value;
    }
    const workLevel = state.workLevel.value ?? state.dziMeta.value?.maxLevel;
    const annotations = Object.entries(bySource)
      .filter(([, ss]) => ss.length > 0)
      .map(([dzip_url, ss]) => ({
        dzip_url,
        level: workLevel,
        strokes: ss.map((s) => {
          const f = workLevel != null ? Math.pow(2, workLevel - s.level) : 1;
          return {
            label: s.labelId,
            points: s.points.map(([x, y]) => [Math.round(x * f), Math.round(y * f)] as [number, number]),
          };
        }),
      }));

    if (annotations.length === 0) {
      msg.value = "No annotations — paint some strokes first.";
      return;
    }

    const fc = state.featureConfig.value;
    msg.value = `Submitting HPC job (${annotations.length} image(s))…`;
    activeJob.value = null;
    logJobId.value = null;
    logText.value = "";

    try {
      const res = await makeClient().submitHeadlessJob({
        annotations,
        features: { filters: featureConfigToFilters(fc), scales: fc.scales },
        p_source: pSource.value.trim(),
        output_dir: outputDir.value.trim(),
      });
      activeJob.value = res;
      msg.value = `Submitted → SLURM ${res.slurm_job_id}`;

      const record: HpcJobRecord = {
        job_id: res.job_id,
        slurm_job_id: res.slurm_job_id,
        slurm_state: res.slurm_state,
        status: res.status,
        p_source: res.p_source,
        output_dir: res.output_dir,
        log_path: res.log_path,
        created_at: Date.now(),
        annotated_images: annotations.length,
      };
      state.addHpcJob(record);
      startPolling(res.job_id);
    } catch (e) {
      msg.value = `Submission failed: ${e}`;
    }
  }

  async function cancelJob() {
    const js = activeJob.value;
    if (!js) return;
    try {
      await makeClient().cancelHeadlessJob(js.job_id);
      stopPolling();
      activeJob.value = { ...js, status: "cancelled" };
      state.updateHpcJob(js.job_id, { status: "cancelled" });
      msg.value = `Cancelled SLURM ${js.slurm_job_id}`;
    } catch (e) {
      msg.value = `Cancel failed: ${e}`;
    }
  }

  async function fetchLog(jobId: string) {
    logJobId.value = jobId;
    logLoading.value = true;
    logText.value = "Loading…";
    try {
      logText.value = await makeClient().getJobLog(jobId, 120);
    } catch (e) {
      logText.value = `Could not fetch log: ${e}`;
    }
    logLoading.value = false;
  }

  function fmtDate(ms: number): string {
    return new Date(ms).toLocaleString(undefined, {
      month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
    });
  }

  const js = activeJob.value;
  const isActive = js && (js.status === "pending" || js.status === "running");
  const health = state.hpcAvailable.value;
  const history = state.hpcJobHistory.value;

  return (
    <section class="panel collapsible">
      <button
        class="collapsible-header"
        onClick={() => {
          open.value = !open.value;
          if (open.value) checkHealth();
        }}
      >
        <span>HPC Export</span>
        <span class="hpc-header-right">
          {health === true && <span class="dot-online" title="Allocator reachable" />}
          {health === false && <span class="dot-offline" title="Allocator unreachable" />}
          {health === null && open.value && <span class="dot-checking" title="Checking…" />}
          {js && <span style={{ marginLeft: 4 }}>{STATUS_ICON[js.status] ?? ""}</span>}
          {history.length > 0 && (
            <span class="job-count-badge">{history.length}</span>
          )}
          <span class="chevron">{open.value ? "▲" : "▼"}</span>
        </span>
      </button>

      {open.value && (
        <div class="collapsible-body">
          {/* Health row */}
          <div class="row" style={{ justifyContent: "space-between" }}>
            <span class="hint">
              Allocator:{" "}
              {health === null ? "checking…" : health ? "✅ reachable" : "❌ unreachable"}
            </span>
            <button class="btn-sm" onClick={checkHealth}>Ping</button>
          </div>

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
            <button class="btn btn-train" onClick={submitJob} disabled={!!isActive}>
              {isActive ? "Running on HPC…" : "Run on HPC"}
            </button>
            {isActive && (
              <button class="btn-sm danger" onClick={cancelJob}>Cancel</button>
            )}
          </div>

          {msg.value && (
            <p class={`status${js?.status === "error" ? " error" : ""}`}>{msg.value}</p>
          )}

          {/* Active job detail + live progress */}
          {js && (
            <div class="hpc-job-info">
              <p class="hint">
                SLURM {js.slurm_job_id} · <strong>{js.slurm_state}</strong>
              </p>

              {/* Live image progress list */}
              {imageProgress.value.length > 0 && (
                <ul class="img-progress-list">
                  {imageProgress.value.map((img) => {
                    const pct = img.tilesTotal > 0
                      ? Math.round((img.tilesDone / img.tilesTotal) * 100)
                      : img.done ? 100 : 0;
                    const statusIcon = img.done ? "✅" : img.failed ? "❌" : "🔄";
                    const shortName = img.name.replace(/\.dzip$/, "");
                    return (
                      <li key={img.idx} class="img-progress-item">
                        <div class="img-progress-header">
                          <span class="img-progress-name" title={img.name}>
                            {statusIcon} {img.idx}/{img.total} {shortName}
                          </span>
                          <span class="img-progress-pct">{pct}%</span>
                        </div>
                        <div class="progress-bar-bg">
                          <div
                            class="progress-bar-fill"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        {!img.done && img.tilesTotal > 0 && (
                          <span class="hint">
                            {img.tilesDone}/{img.tilesTotal} tiles · {img.rate}
                          </span>
                        )}
                      </li>
                    );
                  })}
                </ul>
              )}

              <div class="row" style={{ gap: 6 }}>
                <button
                  class="btn-sm"
                  onClick={() => (showRawLog.value = !showRawLog.value)}
                >
                  {showRawLog.value ? "Hide log" : "Show log"}
                </button>
                <button class="btn-sm" onClick={() => fetchLog(js.job_id)} disabled={logLoading.value}>
                  Refresh log
                </button>
              </div>
              {showRawLog.value && logText.value && (
                <pre class="job-log">{logText.value}</pre>
              )}
            </div>
          )}

          {/* Job history */}
          {history.length > 0 && (
            <div class="divider-row">
              <button
                class="btn-sm"
                style={{ width: "100%" }}
                onClick={() => (showHistory.value = !showHistory.value)}
              >
                {showHistory.value ? "Hide" : "Show"} history ({history.length})
              </button>
            </div>
          )}

          {showHistory.value && (
            <ul class="job-history-list">
              {history.map((r) => (
                <li key={r.job_id} class="job-history-item">
                  <div class="row" style={{ justifyContent: "space-between" }}>
                    <span>{STATUS_ICON[r.status] ?? "?"} SLURM {r.slurm_job_id}</span>
                    <span class="hint">{fmtDate(r.created_at)}</span>
                  </div>
                  <span class="hint">{r.annotated_images} img · {r.slurm_state}</span>
                  <button
                    class="btn-sm"
                    onClick={() => fetchLog(r.job_id)}
                    disabled={logLoading.value}
                  >
                    {logJobId.value === r.job_id ? "Refresh" : "Log"}
                  </button>
                  {logJobId.value === r.job_id && logText.value && (
                    <pre class="job-log">{logText.value}</pre>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}
