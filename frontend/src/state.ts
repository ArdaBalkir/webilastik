import { signal, computed } from "@preact/signals";
import type { Label, Stroke, FeatureConfig, DziMeta, Project } from "./types";
import { DEFAULT_FEATURE_CONFIG } from "./types";

// ── App-level reactive state ─────────────────────────────────────────────────

export const serverUrl = signal<string>(
  import.meta.env.VITE_COMPUTE_SERVER_URL ?? "http://localhost:8000",
);
export const bearerToken = signal<string>("");

// Image / project
export const dziUrl = signal<string>("");
export const dziMeta = signal<DziMeta | null>(null);
export const dziName = signal<string>("");
export const loadError = signal<string>("");
export const isLoadingImage = signal<boolean>(false);

// Labels
export const labels = signal<Label[]>([
  { id: 1, name: "Foreground", color: "#e05252" },
  { id: 2, name: "Background", color: "#6699ff" },
]);
export const activeLabelId = signal<number>(1);

// Strokes
export const strokes = signal<Stroke[]>([]);

// Tool mode
export const toolMode = signal<"pan" | "brush">("pan");
export const brushSize = signal<number>(3);

// Features
export const featureConfig = signal<FeatureConfig>({
  ...DEFAULT_FEATURE_CONFIG,
});

// Training
export const trainingStatus = signal<"idle" | "training" | "ready" | "error">(
  "idle",
);
export const trainingError = signal<string>("");
export const classifierId = signal<string | null>(null);
export const numClasses = signal<number>(0);

// Prediction overlay
export const predictionVisible = signal<boolean>(true);
export const predictionOpacity = signal<number>(0.5);

// Export
export const exportStatus = signal<string>("");
export const exportJobId = signal<string | null>(null);

// Working level — null = use maxLevel (full res)
export const workLevelOffset = signal<number>(0); // 0 = full res, 1 = half res, etc.
export const workLevel = computed(() => {
  const meta = dziMeta.value;
  if (!meta) return null;
  return Math.max(0, meta.maxLevel - workLevelOffset.value);
});

// The DZI level that the current classifier was trained at
export const trainedLevel = signal<number | null>(null);

// ── Project serialization ────────────────────────────────────────────────────

export function saveProject(): void {
  const project: Project = {
    dziUrl: dziUrl.value,
    dziName: dziName.value,
    workLevel: workLevel.value,
    labels: labels.value,
    strokes: strokes.value,
  };
  const json = JSON.stringify(project, null, 2);
  const blob = new Blob([json], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "project.json";
  a.click();
  URL.revokeObjectURL(url);
}

export function loadProject(file: File): Promise<void> {
  return file.text().then((text) => {
    const project = JSON.parse(text) as Project;
    dziUrl.value = project.dziUrl;
    dziName.value = project.dziName;
    labels.value = project.labels;
    strokes.value = project.strokes;
    classifierId.value = null;
    trainingStatus.value = "idle";
  });
}

export function nextLabelId(): number {
  const max = labels.value.reduce(
    (m: number, l: Label) => Math.max(m, l.id),
    0,
  );
  return max + 1;
}
