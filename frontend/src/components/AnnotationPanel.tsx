import { h } from "preact";
import { useSignal } from "@preact/signals";
import { Trash2, X } from "lucide-preact";
import * as state from "../state";

export function AnnotationPanel() {
  const open = useSignal(false);

  const strokes = state.strokes.value;
  const labels = state.labels.value;
  const total = strokes.length;

  // Group strokes by label, keeping the original index for delete/highlight
  const groups = labels
    .map((label) => ({
      label,
      entries: strokes
        .map((s, i) => ({ s, i }))
        .filter(({ s }) => s.labelId === label.id),
    }))
    .filter((g) => g.entries.length > 0);

  return (
    <div class="subsection collapsible">
      <button
        class="subsection-toggle"
        onClick={() => (open.value = !open.value)}
      >
        <span>Annotations</span>
        <span class="ann-header-right">
          <span class="hint">
            {total} stroke{total !== 1 ? "s" : ""}
          </span>
          {total > 0 && (
            <button
              class="btn-icon"
              title="Clear all strokes"
              onClick={(e) => {
                e.stopPropagation();
                state.clearAllStrokes();
              }}
            >
              <Trash2 size={13} />
            </button>
          )}
          <span class="chevron">{open.value ? "▲" : "▼"}</span>
        </span>
      </button>

      {open.value && (
        <div class="collapsible-body ann-body">
          {total === 0 ? (
            <p class="hint" style={{ textAlign: "center", padding: "8px 0" }}>
              No strokes yet — use the brush tool.
            </p>
          ) : (
            groups.map(({ label, entries }) => (
              <div key={label.id} class="ann-group">
                <div class="ann-group-header">
                  <span class="ann-group-title">
                    <span
                      class="ann-label-dot"
                      style={{ background: label.color }}
                    />
                    <span>{label.name}</span>
                    <span class="hint">({entries.length})</span>
                  </span>
                  <button
                    class="btn-sm danger"
                    title={`Clear all ${label.name} strokes`}
                    onClick={() => state.clearStrokesByLabel(label.id)}
                  >
                    Clear
                  </button>
                </div>

                <ul class="ann-list">
                  {entries.map(({ s, i }) => (
                    <li
                      key={i}
                      class="ann-stroke-item"
                      onMouseEnter={() =>
                        (state.highlightedStrokeIdx.value = i)
                      }
                      onMouseLeave={() =>
                        (state.highlightedStrokeIdx.value = null)
                      }
                    >
                      <span class="hint">
                        #{i + 1} · {s.points.length} pts
                      </span>
                      <button
                        class="btn-icon"
                        title="Delete this stroke"
                        onClick={() => state.deleteStrokeAt(i)}
                      >
                        <X size={12} />
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
