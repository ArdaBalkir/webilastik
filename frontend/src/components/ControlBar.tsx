import { h } from "preact";
import * as state from "../state";
import { ApiClient, featureConfigToFilters } from "../api";
import type { ExportRequest } from "../types";
import { Hand, Paintbrush, Download, Upload } from "lucide-preact";

interface Props {
  onTrain: () => Promise<void>;
}

export function ControlBar({ onTrain }: Props) {
  const tool = state.toolMode.value;
  const trainStatus = state.trainingStatus.value;
  const predVisible = state.predictionVisible.value;
  const opacity = state.predictionOpacity.value;
  const brushSz = state.brushSize.value;
  const strokes = state.strokes.value;

  async function handleExport() {
    const cid = state.classifierId.value;
    const meta = state.dziMeta.value;
    if (!cid || !meta) return;

    // Derive the output filename from the source DZIP URL
    const srcUrl = state.dziUrl.value;
    const srcFilename =
      srcUrl.split("/").pop()?.split("?")[0] ?? "predictions.dzip";

    const dest =
      prompt(
        `Output filename will be: ${srcFilename}\n\n` +
          `Leave blank to download locally.\n` +
          `Or enter a directory/bucket URL to upload there:\n` +
          `e.g. https://data-proxy.ebrains.eu/api/v1/buckets/my-bucket`,
      ) ?? "";

    const fc = state.featureConfig.value;
    const req: ExportRequest = {
      classifier_id: cid,
      dzip_url: state.dziUrl.value,
      dzi_name: state.dziName.value,
      level: state.workLevel.value ?? meta.maxLevel,
      features: { filters: featureConfigToFilters(fc), scales: fc.scales },
      ...(dest.trim() ? { output_url: dest.trim() } : {}),
    };

    const client = new ApiClient(
      state.serverUrl.value,
      state.bearerToken.value,
    );

    try {
      if (dest.trim()) {
        // Upload tile-by-tile via polling endpoint (avoids 413 size limit)
        state.exportStatus.value = `Submitting upload job…`;
        const { job_id } = await client.startExport(req);
        state.exportJobId.value = job_id;
        pollExport(job_id, client);
      } else {
        // Browser download as DZIP
        state.exportStatus.value = "Building DZIP…";
        await client.exportZipDownload(req);
        state.exportStatus.value = "Downloaded!";
      }
    } catch (err) {
      state.exportStatus.value = `Error: ${err}`;
    }
  }

  return (
    <section class="panel">
      <h2>Tools</h2>

      <div class="tool-row">
        <button
          class={`btn-tool ${tool === "pan" ? "active" : ""}`}
          onClick={() => (state.toolMode.value = "pan")}
          title="Pan / Zoom"
        >
          <Hand size={14} /> Pan
        </button>
        <button
          class={`btn-tool ${tool === "brush" ? "active" : ""}`}
          onClick={() => (state.toolMode.value = "brush")}
          title="Brush to annotate"
        >
          <Paintbrush size={14} /> Brush
        </button>
      </div>

      {tool === "brush" && (
        <label class="row">
          Brush radius:
          <input
            type="range"
            min={1}
            max={40}
            value={brushSz}
            onInput={(e) =>
              (state.brushSize.value = parseInt(
                (e.target as HTMLInputElement).value,
              ))
            }
          />
          <span>{brushSz} px</span>
        </label>
      )}

      <div class="row">
        <span>
          {strokes.length} stroke{strokes.length !== 1 ? "s" : ""}
        </span>
        {strokes.length > 0 && (
          <button
            class="btn-sm danger"
            onClick={() => (state.strokes.value = [])}
          >
            Clear
          </button>
        )}
      </div>

      <button
        class={`btn btn-train ${trainStatus === "training" ? "loading" : ""}`}
        onClick={onTrain}
        disabled={trainStatus === "training" || strokes.length === 0}
      >
        {trainStatus === "training" ? "Training…" : "Train classifier"}
      </button>

      {trainStatus === "error" && (
        <p class="error">{state.trainingError.value}</p>
      )}

      {trainStatus === "ready" && (
        <>
          <div class="row">
            <label>
              <input
                type="checkbox"
                checked={predVisible}
                onChange={(e) =>
                  (state.predictionVisible.value = (
                    e.target as HTMLInputElement
                  ).checked)
                }
              />
              Show predictions
            </label>
          </div>
          {predVisible && (
            <label class="row">
              Opacity:
              <input
                type="range"
                min={0}
                max={100}
                value={Math.round(opacity * 100)}
                onInput={(e) =>
                  (state.predictionOpacity.value =
                    parseInt((e.target as HTMLInputElement).value) / 100)
                }
              />
            </label>
          )}
          <button class="btn-sm" onClick={handleExport}>
            <Download
              size={11}
              style="vertical-align:middle;margin-right:3px"
            />
            Export…
          </button>
          {state.exportStatus.value && (
            <p class="status">{state.exportStatus.value}</p>
          )}
        </>
      )}
    </section>
  );
}

function pollExport(jobId: string, client: ApiClient) {
  let tries = 0;
  const interval = setInterval(async () => {
    tries++;
    try {
      const s = await client.getExportStatus(jobId);
      if (s.status === "done") {
        state.exportStatus.value = "Export complete!";
        clearInterval(interval);
      } else if (s.status === "error") {
        state.exportStatus.value = `Export error: ${s.error ?? "unknown"}`;
        clearInterval(interval);
      } else {
        const pct = s.progress ? ` (${Math.round(s.progress * 100)}%)` : "";
        state.exportStatus.value = `Exporting${pct}…`;
      }
    } catch {
      if (tries > 100) clearInterval(interval);
    }
  }, 3000);
}
