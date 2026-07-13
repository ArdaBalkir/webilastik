import { h } from "preact";
import { useSignal } from "@preact/signals";
import * as state from "../state";
import { Hand, Paintbrush, Eraser } from "lucide-preact";
import { LabelPanel } from "./LabelPanel";
import { AnnotationPanel } from "./AnnotationPanel";

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
  const canTrain = state.totalAnnotatedStrokes.value > 0 && trainStatus !== "training";

  const saveModelName = useSignal("");
  const showSaved = useSignal(false);

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
        <button
          class={`btn-tool ${tool === "erase" ? "active" : ""}`}
          onClick={() => (state.toolMode.value = "erase")}
          title="Eraser — click/drag over strokes to remove them"
        >
          <Eraser size={14} /> Erase
        </button>
      </div>

      {(tool === "brush" || tool === "erase") && (
        <label class="row">
          {tool === "erase" ? "Eraser radius:" : "Brush radius:"}
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

      <LabelPanel />
      <AnnotationPanel />

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
            {state.exportStatus.value === "submitting" ? "Submitting…" : "Segment all images"}
          </button>
          {state.exportStatus.value && state.exportStatus.value !== "submitting" && (
            <p class="status">{state.exportStatus.value}</p>
          )}

          {/* ── Save this model ─────────────────────────────────── */}
          <div class="subsection">
            <div class="subsection-title">Save model</div>
            <div class="row" style="gap:4px">
              <input
                class="input-sm"
                style="flex:1"
                placeholder="Model name…"
                value={saveModelName.value}
                onInput={(e) =>
                  (saveModelName.value = (e.target as HTMLInputElement).value)
                }
              />
              <button
                class="btn-sm"
                disabled={!saveModelName.value.trim()}
                onClick={() => {
                  state.saveCurrentModel(saveModelName.value.trim());
                  saveModelName.value = "";
                }}
              >
                Save
              </button>
            </div>
          </div>
        </>
      )}

      {/* ── Saved models ──────────────────────────────────────────── */}
      {state.savedModels.value.length > 0 && (
        <div class="subsection">
          <button
            class="subsection-title"
            style="background:none;border:none;color:inherit;cursor:pointer;text-align:left;padding:0;width:100%"
            onClick={() => (showSaved.value = !showSaved.value)}
          >
            Saved models ({state.savedModels.value.length}){" "}
            {showSaved.value ? "▾" : "▸"}
          </button>
          {showSaved.value && (
            <ul class="saved-model-list">
              {state.savedModels.value.map((m) => (
                <li key={m.id} class="saved-model-item">
                  <div class="saved-model-name">{m.name}</div>
                  <div class="saved-model-meta muted">
                    {m.numClasses != null ? `${m.numClasses} classes · ` : ""}
                    {new Date(m.savedAt).toLocaleDateString()}
                  </div>
                  <div class="row" style="gap:4px;margin-top:3px">
                    <button
                      class="btn-sm"
                      title="Load this model — predictions appear immediately"
                      onClick={() => state.restoreModel(m)}
                    >
                      Load
                    </button>
                    <button
                      class="btn-sm btn-danger"
                      onClick={() => state.deleteSavedModel(m.id)}
                    >
                      ✕
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}
