/**
 * ExportPanel — batch prediction export collapsible
 *
 * Reads ?p_source= and ?output_dir= from state (URL params).
 * Resolves all DZIPs in p_source, sends a batch-export job per classifier,
 * and polls progress.
 */
import { h } from "preact";
import { useEffect } from "preact/hooks";
import { useSignal } from "@preact/signals";
import { ApiClient, featureConfigToFilters } from "../api";
import * as state from "../state";
import type { BatchExportStatus } from "../types";

export function ExportPanel() {
  const open = useSignal(false);
  const pSource = useSignal(state.pSourceUrl.value);
  const outputDir = useSignal(state.outputDirUrl.value);
  const msg = useSignal("");
  const jobId = state.batchJobId;
  const batchStatus = state.batchStatus;

  // Poll when a job is running
  useEffect(() => {
    let interval: ReturnType<typeof setInterval> | null = null;

    const unsub = jobId.subscribe((id) => {
      if (interval) clearInterval(interval);
      if (!id) return;
      const client = new ApiClient(
        state.serverUrl.value,
        state.bearerToken.value,
      );
      interval = setInterval(async () => {
        try {
          const s = await client.getBatchExportStatus(id);
          batchStatus.value = s;
          if (s.status === "done" || s.status === "error") {
            clearInterval(interval!);
            interval = null;
            msg.value =
              s.status === "done"
                ? `Done — ${s.done}/${s.total} exported.`
                : `Error: ${s.error ?? "unknown"}`;
          }
        } catch (e) {
          msg.value = String(e);
          clearInterval(interval!);
        }
      }, 2500);
    });

    return () => {
      unsub();
      if (interval) clearInterval(interval);
    };
  }, []);

  async function startExport() {
    const cid = state.classifierId.value;
    if (!cid) {
      msg.value = "Train a classifier first.";
      return;
    }
    if (!pSource.value.trim() || !outputDir.value.trim()) {
      msg.value = "Set both p_source and output_dir.";
      return;
    }
    msg.value = "Submitting batch job…";
    batchStatus.value = null;
    try {
      const client = new ApiClient(
        state.serverUrl.value,
        state.bearerToken.value,
      );
      const fc = state.featureConfig.value;
      const res = await client.startBatchExport({
        classifier_id: cid,
        p_source: pSource.value.trim(),
        output_dir: outputDir.value.trim(),
        features: {
          filters: featureConfigToFilters(fc),
          scales: fc.scales,
        },
      });
      jobId.value = res.job_id;
      msg.value = `Job started: ${res.job_id}`;
    } catch (e) {
      msg.value = `Failed: ${e}`;
    }
  }

  const status = batchStatus.value;

  return (
    <section class="panel collapsible">
      <button
        class="collapsible-header"
        onClick={() => (open.value = !open.value)}
      >
        <span>Batch Export</span>
        <span class="chevron">{open.value ? "▲" : "▼"}</span>
      </button>

      {open.value && (
        <div class="collapsible-body">
          <label class="hint">Source directory (p_source)</label>
          <input
            class="input-url"
            value={pSource.value}
            placeholder="https://data-proxy.ebrains.eu/api/v1/buckets/…"
            onInput={(e: Event) =>
              (pSource.value = (e.target as HTMLInputElement).value)
            }
          />
          <label class="hint">Output directory</label>
          <input
            class="input-url"
            value={outputDir.value}
            placeholder="https://data-proxy.ebrains.eu/api/v1/buckets/…/segmentations"
            onInput={(e: Event) =>
              (outputDir.value = (e.target as HTMLInputElement).value)
            }
          />

          <button
            class="btn btn-train"
            onClick={startExport}
            disabled={!state.classifierId.value}
          >
            Run Batch Export
          </button>

          {msg.value && <p class="status">{msg.value}</p>}

          {status && (
            <div class="batch-progress">
              <div class="progress-bar-bg">
                <div
                  class="progress-bar-fill"
                  style={{
                    width: `${Math.round((status.progress ?? 0) * 100)}%`,
                  }}
                />
              </div>
              <p class="hint">
                {status.done}/{status.total} images
                {status.current ? ` — ${status.current}` : ""}
              </p>
              {status.failed.length > 0 && (
                <p class="status error">
                  {status.failed.length} failed:{" "}
                  {status.failed
                    .slice(0, 3)
                    .map((f) => f.name)
                    .join(", ")}
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
