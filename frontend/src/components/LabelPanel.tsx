import { h } from "preact";
import * as state from "../state";
import { nextLabelId } from "../state";
import type { Label } from "../types";
import { X, Plus } from "lucide-preact";

const PALETTE = [
  "#e05252",
  "#52aee0",
  "#52e06b",
  "#e0c452",
  "#a052e0",
  "#e08852",
  "#52e0d4",
  "#e052b8",
  "#7de052",
  "#5265e0",
];

export function LabelPanel() {
  const labels = state.labels.value;
  const activeId = state.activeLabelId.value;

  function addLabel() {
    const id = nextLabelId();
    const color = PALETTE[labels.length % PALETTE.length];
    state.labels.value = [...labels, { id, name: `Label ${id}`, color }];
    state.activeLabelId.value = id;
  }

  function updateLabel(id: number, patch: Partial<Label>) {
    state.labels.value = labels.map((l) =>
      l.id === id ? { ...l, ...patch } : l,
    );
  }

  function removeLabel(id: number) {
    state.labels.value = labels.filter((l) => l.id !== id);
    state.strokes.value = state.strokes.value.filter((s) => s.labelId !== id);
    if (state.activeLabelId.value === id) {
      state.activeLabelId.value = labels[0]?.id ?? 1;
    }
  }

  return (
    <section class="panel">
      <h2>Labels</h2>
      <ul class="label-list">
        {labels.map((label) => (
          <li
            key={label.id}
            class={`label-item ${activeId === label.id ? "active" : ""}`}
            onClick={() => (state.activeLabelId.value = label.id)}
          >
            <input
              type="color"
              value={label.color}
              class="color-swatch"
              onInput={(e) =>
                updateLabel(label.id, {
                  color: (e.target as HTMLInputElement).value,
                })
              }
              onClick={(e) => e.stopPropagation()}
            />
            <input
              class="label-name"
              type="text"
              value={label.name}
              onInput={(e) =>
                updateLabel(label.id, {
                  name: (e.target as HTMLInputElement).value,
                })
              }
              onClick={(e) => e.stopPropagation()}
            />
            {labels.length > 1 && (
              <button
                class="btn-icon"
                onClick={(e) => {
                  e.stopPropagation();
                  removeLabel(label.id);
                }}
                title="Remove label"
              >
                <X size={13} />
              </button>
            )}
          </li>
        ))}
      </ul>
      <button class="btn-sm" onClick={addLabel}>
        <Plus size={11} style="vertical-align:middle;margin-right:2px" />
        Add label
      </button>
    </section>
  );
}
