import { h } from "preact";
import * as state from "../state";
import { Hand, Paintbrush } from "lucide-preact";

interface Props {
  onTrain: () => Promise<void>;
  onExport: () => Promise<void>;
}

export function ControlBar({ onTrain, onExport }: Props) {
  const tool = state.toolMode.value;
  const trainStatus = state.trainingStatus.value;
  const predVisible = state.predictionVisible.value;
  const opacity = state.predictionOpacity.value;
  const brushSz = state.brushSize.value;
  const strokes = state.strokes.value;
  const canTrain = strokes.length > 0 && trainStatus !== "training";

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
        <span>{strokes.length} stroke{strokes.length !== 1 ? "s" : ""}</span>
        {strokes.length > 0 && (
          <button class="btn-sm danger" onClick={() => (state.strokes.value = [])}>
            Clear
          </button>
        )}
      </div>

      <button
        class={`btn btn-train ${trainStatus === "training" ? "loading" : ""}`}
        onClick={onTrain}
        disabled={!canTrain}
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
              <span>{Math.round(opacity * 100)}%</span>
            </label>
          )}

          <button
            class="btn btn-export"
            onClick={onExport}
            disabled={state.exportStatus.value === "submitting"}
          >
            {state.exportStatus.value === "submitting" ? "Submitting…" : "Export to HPC"}
          </button>
          {state.exportStatus.value && state.exportStatus.value !== "submitting" && (
            <p class="status">{state.exportStatus.value}</p>
          )}
        </>
      )}
    </section>
  );
}
