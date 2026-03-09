import { h } from "preact";
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
  const fc = state.featureConfig.value;

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
    <section class="panel">
      <h2>Features</h2>
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
    </section>
  );
}
