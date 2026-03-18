import { signal, computed } from "@preact/signals";
import type {
  Label,
  Stroke,
  FeatureConfig,
  DziMeta,
  Project,
  SourceEntry,
  BatchExportStatus,
} from "./types";
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

// ── URL param-seeded values (set once at startup by app.tsx) ───────────────
export const tSourceUrl = signal<string>(""); // ?t_source= training dir / DZIP URL
export const pSourceUrl = signal<string>(""); // ?p_source= export dir URL
export const outputDirUrl = signal<string>(""); // ?output_dir= destination dir URL

// Allocator URL — points at session_allocator.py (port 8001 by default)
export const allocatorUrl = signal<string>(
  import.meta.env.VITE_ALLOCATOR_URL ?? "http://localhost:8001",
);

// ── Training source browser ───────────────────────────────────────────────
export const trainingSources = signal<SourceEntry[]>([]);
export const selectedTrainingSources = signal<Set<string>>(new Set()); // object_url set
export const trainingSourcesLoading = signal<boolean>(false);
export const trainingSourcesError = signal<string>("");

// Per-image strokes keyed by object_url — persists annotations across image switches
export const strokesBySource = signal<Record<string, Stroke[]>>({});

/** Switch the active image for annotation. Saves + restores strokes. */
export function switchTrainingSource(objectUrl: string) {
  // Save current strokes under current dziUrl
  if (dziUrl.value) {
    strokesBySource.value = {
      ...strokesBySource.value,
      [dziUrl.value]: strokes.value,
    };
  }
  // Restore strokes for new source
  strokes.value = strokesBySource.value[objectUrl] ?? [];
}

// ── Batch export state ────────────────────────────────────────────────────
export const batchJobId = signal<string | null>(null);
export const batchStatus = signal<BatchExportStatus | null>(null);

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

/**
 * Export all annotations (across all training images) in the format
 * expected by headless_cli.py --annotations and POST /headless-run.
 *
 * Points are rescaled to max_level (full resolution) so the CLI does not
 * need to know which zoom level was active when strokes were captured.
 */
export function saveAnnotationsForHeadless(): void {
  // Flush current image strokes into the per-source map
  const current: Record<string, Stroke[]> = {
    ...strokesBySource.value,
  };
  if (dziUrl.value && strokes.value.length > 0) {
    current[dziUrl.value] = strokes.value;
  }

  const meta = dziMeta.value;
  const targetLevel = meta ? meta.maxLevel : null;

  const annotations = Object.entries(current)
    .filter(([, ss]) => ss.length > 0)
    .map(([dzip_url, ss]) => ({
      dzip_url,
      strokes: ss.map((s) => ({
        label: s.labelId,
        points: s.points.map(([x, y]) => {
          // Rescale from stroke capture level → full res (max_level)
          const factor =
            targetLevel !== null ? Math.pow(2, targetLevel - s.level) : 1;
          return [Math.round(x * factor), Math.round(y * factor)];
        }),
      })),
    }));

  if (annotations.length === 0) {
    alert("No annotations to export — paint some strokes first.");
    return;
  }

  const json = JSON.stringify(annotations, null, 2);
  const blob = new Blob([json], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  // e.g. "annotations_2img.json"
  a.download = `annotations_${annotations.length}img.json`;
  a.click();
  URL.revokeObjectURL(url);
}

export function nextLabelId(): number {
  const max = labels.value.reduce(
    (m: number, l: Label) => Math.max(m, l.id),
    0,
  );
  return max + 1;
}
