import { signal, computed } from "@preact/signals";
import type {
  Label,
  Stroke,
  FeatureConfig,
  DziMeta,
  Project,
  SourceEntry,
  BatchExportStatus,
  HpcJobRecord,
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
// ?workdir=https://…/bucket/project/  →  source= workdir/zipped_images/
//                                         output= workdir/segmentations/
export const workdir = signal<string>("");
export const sourceDir = computed(() => {
  const w = workdir.value.replace(/\/$/, "");
  return w ? `${w}/zipped_images/` : "";
});
export const outputDir = computed(() => {
  const w = workdir.value.replace(/\/$/, "");
  return w ? `${w}/segmentations/` : "";
});

// Kept for backward-compat with old ?t_source= / ?p_source= params
export const tSourceUrl = signal<string>("");
export const pSourceUrl = signal<string>("");
export const outputDirUrl = signal<string>("");

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

/** Total stroke count across all annotated images (including current). */
export const totalAnnotatedStrokes = computed(() => {
  const bySource = strokesBySource.value;
  const current = strokes.value;
  const currentUrl = dziUrl.value;
  let total = current.length;
  for (const [url, ss] of Object.entries(bySource)) {
    if (url !== currentUrl) total += ss.length;
  }
  return total;
});

/** Number of distinct images with at least one stroke. */
export const annotatedImageCount = computed(() => {
  const bySource = strokesBySource.value;
  const currentUrl = dziUrl.value;
  const hasCurrentStrokes = strokes.value.length > 0;
  const set = new Set(
    Object.entries(bySource)
      .filter(([url, ss]) => ss.length > 0 && url !== currentUrl)
      .map(([url]) => url),
  );
  if (hasCurrentStrokes && currentUrl) set.add(currentUrl);
  return set.size;
});

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

// ── HPC job history (persisted to localStorage) ───────────────────────────
const _JOB_HISTORY_KEY = "wi2_hpc_jobs";

function _loadJobHistory(): HpcJobRecord[] {
  try {
    return JSON.parse(localStorage.getItem(_JOB_HISTORY_KEY) ?? "[]");
  } catch {
    return [];
  }
}

export const hpcJobHistory = signal<HpcJobRecord[]>(_loadJobHistory());

// Restore active job from history on startup (most recent non-terminal job)
const _TERMINAL = new Set(["done", "error", "cancelled"]);
const _restoredJob = _loadJobHistory().find((r) => !_TERMINAL.has(r.status)) ?? null;
export const activeHpcJob = signal<HpcJobRecord | null>(_restoredJob);

export function addHpcJob(record: HpcJobRecord): void {
  const next = [record, ...hpcJobHistory.value].slice(0, 20); // keep last 20
  hpcJobHistory.value = next;
  localStorage.setItem(_JOB_HISTORY_KEY, JSON.stringify(next));
}

export function updateHpcJob(jobId: string, patch: Partial<HpcJobRecord>): void {
  const next = hpcJobHistory.value.map((r) =>
    r.job_id === jobId ? { ...r, ...patch } : r,
  );
  hpcJobHistory.value = next;
  localStorage.setItem(_JOB_HISTORY_KEY, JSON.stringify(next));
}

// ── HPC / allocator availability ──────────────────────────────────────────
export const hpcAvailable = signal<boolean | null>(null); // null = unknown

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

// ── Annotation management ────────────────────────────────────────────────────

/** Index of the stroke currently hovered in the Annotations panel (for canvas highlight). */
export const highlightedStrokeIdx = signal<number | null>(null);

/** Delete the stroke at the given index in the current image's stroke list. */
export function deleteStrokeAt(idx: number): void {
  strokes.value = strokes.value.filter((_, i) => i !== idx);
  highlightedStrokeIdx.value = null;
}

/** Remove all strokes for a given label from the current image. */
export function clearStrokesByLabel(labelId: number): void {
  strokes.value = strokes.value.filter((s) => s.labelId !== labelId);
  highlightedStrokeIdx.value = null;
}

/** Clear every stroke on the current image. */
export function clearAllStrokes(): void {
  strokes.value = [];
  highlightedStrokeIdx.value = null;
}
