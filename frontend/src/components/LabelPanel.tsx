import { h } from "preact";
import * as state from "../state";

export function LabelPanel() {
  const labels = state.labels.value;
  const activeId = state.activeLabelId.value;

  return (
    <div class="subsection">
      <h3 class="subsection-title">Labels</h3>
      <ul class="label-list">
        {labels.map((label) => (
          <li key={label.id}>
            <button
              type="button"
              class={`label-item ${activeId === label.id ? "active" : ""}`}
              aria-pressed={activeId === label.id}
              onClick={() => (state.activeLabelId.value = label.id)}
            >
              <span
                class="label-color"
                style={{ background: label.color }}
                aria-hidden="true"
              />
              <span class="label-text">{label.name}</span>
              {activeId === label.id && (
                <span class="label-active-indicator">Selected</span>
              )}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
