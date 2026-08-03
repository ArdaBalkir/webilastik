import { h } from "preact";
import { useSignal } from "@preact/signals";
import * as state from "../state";
import type { FeatureConfig } from "../types";

const FILTER_LABELS: Record<keyof Omit<FeatureConfig, "scales">, string> = {
  gaussianSmoothing: "Gaussian Smoothing",
  laplacianOfGaussian: "Laplacian of Gaussian",
  gaussianGradientMagnitude: "Gradient Magnitude",
  differenceOfGaussians: "Diff. of Gaussians",
  structureTensorEigenvalues: "Structure Tensor Eigenvalues",
  hessianOfGaussianEigenvalues: "Hessian of Gaussian Eigenvalues",
};

const DEFAULT_SCALES = [0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0];

export function FeaturePanel() {
  const open = useSignal(false);
  const fc = state.featureConfig.value;

  const enabledCount = (Object.keys(FILTER_LABELS) as Array<keyof typeof FILTER_LABELS>)
    .filter((k) => fc[k]).length;

  function toggle(key: keyof Omit<FeatureConfig, "scales">) {
    state.featureConfig.value = { ...fc, [key]: !fc[key] };
  }

  function toggleScale(s: number) {
    const cur = fc.scales;
    state.featureConfig.value = {
      ...fc,
      scales: cur.includes(s)
        ? cur.filter((x) => x !== s)
        : [...cur, s].sort((a, b) => a - b),
    };
  }

  return (
    <section class="panel collapsible">
      <button
        type="button"
        class="collapsible-header"
        aria-expanded={open.value}
        aria-controls="feature-panel-body"
        onClick={() => (open.value = !open.value)}
      >
        <span>Features</span>
        <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span class="hint">{enabledCount} filters · {fc.scales.length} σ</span>
          <span class="chevron">{open.value ? "▲" : "▼"}</span>
        </span>
      </button>

      {open.value && (
        <div id="feature-panel-body" class="collapsible-body">
          <div class="feature-filters">
            {(Object.keys(FILTER_LABELS) as Array<keyof typeof FILTER_LABELS>).map(
              (k) => (
                <label key={k} class="checkbox-label">
                  <input
                    type="checkbox"
                    checked={fc[k] as boolean}
                    onChange={() => toggle(k)}
                  />
                  {FILTER_LABELS[k]}
                </label>
              ),
            )}
          </div>
          <div class="scales-section">
            <span class="label-sm">Scales (σ):</span>
            <div class="scales-chips">
              {DEFAULT_SCALES.map((s) => (
                <button
                  key={s}
                  class={`chip ${fc.scales.includes(s) ? "active" : ""}`}
                  onClick={() => toggleScale(s)}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
