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
  { id: 1, name: "Foreground", color: "#ff0000" },
  { id: 2, name: "Background", color: "#6699ff" },
]);
export const activeLabelId = signal<number>(1);

// Strokes
export const strokes = signal<Stroke[]>([]);

// Tool mode
export const toolMode = signal<"pan" | "brush" | "erase">("pan");
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
export const workLevelOffset = signal<number>(2); // 0 = full res, 1 = half, 2 = quarter (25%)
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

// Quick-switch list: images the user has pinned for fast access
export const pinnedSources = signal<SourceEntry[]>([]);

/** Add entries to pinnedSources, skipping duplicates. */
export function pinSources(entries: SourceEntry[]): void {
  const existing = new Set(pinnedSources.value.map((s) => s.object_url));
  const toAdd = entries.filter((e) => !existing.has(e.object_url));
  if (toAdd.length > 0) pinnedSources.value = [...pinnedSources.value, ...toAdd];
}

/** Remove a single entry from the pinned list. */
export function unpinSource(objectUrl: string): void {
  pinnedSources.value = pinnedSources.value.filter(
    (s) => s.object_url !== objectUrl,
  );
}

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
  if (dziUrl.value) {
    strokesBySource.value = { ...strokesBySource.value, [dziUrl.value]: strokes.value };
  }
  // Update dziUrl immediately so computed signals (totalAnnotatedStrokes) stay
  // consistent during the async image load that follows.
  dziUrl.value = objectUrl;
  strokes.value = strokesBySource.value[objectUrl] ?? [];
}

// ── Saved classifier models (persisted to localStorage) ───────────────────
// Users can name and save a classifier_id so it survives page reload.
// On load they can pick a saved model, enter it as the active classifierId,
// and predictions appear immediately without re-training.

export interface SavedModel {
  id: string;           // classifier_id from the server
  name: string;         // user-supplied label
  savedAt: number;      // Date.now()
  dziUrl?: string;      // which image it was trained on (for reference)
  numClasses?: number;
}

const _MODEL_STORE_KEY = "wi2_saved_models";

function _loadSavedModels(): SavedModel[] {
  try { return JSON.parse(localStorage.getItem(_MODEL_STORE_KEY) ?? "[]"); }
  catch { return []; }
}

export const savedModels = signal<SavedModel[]>(_loadSavedModels());

export function saveCurrentModel(name: string): void {
  const id = classifierId.value;
  if (!id) return;
  const rec: SavedModel = {
    id,
    name,
    savedAt: Date.now(),
    dziUrl: dziUrl.value || undefined,
    numClasses: numClasses.value || undefined,
  };
  const next = [rec, ...savedModels.value.filter((m) => m.id !== id)].slice(0, 20);
  savedModels.value = next;
  localStorage.setItem(_MODEL_STORE_KEY, JSON.stringify(next));
}

export function deleteSavedModel(id: string): void {
  const next = savedModels.value.filter((m) => m.id !== id);
  savedModels.value = next;
  localStorage.setItem(_MODEL_STORE_KEY, JSON.stringify(next));
}

/** Restore a saved model as the active classifier (no re-training needed). */
export function restoreModel(model: SavedModel): void {
  classifierId.value   = model.id;
  numClasses.value     = model.numClasses ?? 0;
  trainingStatus.value = "ready";
  predictionVisible.value = true;
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

export async function saveProjectToCloud(): Promise<string> {
  // Flush current image strokes into the per-source map before saving
  const allBySource: Record<string, Stroke[]> = { ...strokesBySource.value };
  if (dziUrl.value) {
    allBySource[dziUrl.value] = strokes.value;
  }
  const project: Project = {
    dziUrl: dziUrl.value,
    dziName: dziName.value,
    workLevel: workLevel.value,
    labels: labels.value,
    strokes: strokes.value,
    strokesBySource: allBySource,
  };
  const json = JSON.stringify(project, null, 2);

  const currentUrl = dziUrl.value;
  const token = bearerToken.value;
  const server = serverUrl.value;
  if (!currentUrl || !token || !server) {
    throw new Error("Open a data-proxy image with a bearer token first.");
  }
  const match = currentUrl.match(/\/v1\/buckets\/([^/]+)\//);
  if (!match) {
    throw new Error("Could not determine bucket from the current image URL.");
  }
  const bucket = match[1];
  const ts = new Date().toISOString().replace(/[:.]/g, "-");
  const destUrl =
    `https://data-proxy.ebrains.eu/api/v1/buckets/${bucket}` +
    `/ilastikProjectSaves/project_${ts}.json`;
  const { ApiClient } = await import("./api");
  const res = await new ApiClient(server, token).saveProjectToProxy(json, destUrl);
  return res.url;
}

export function loadProject(file: File): Promise<void> {
  return file.text().then((text) => {
    const project = JSON.parse(text) as Project;
    dziUrl.value = project.dziUrl;
    dziName.value = project.dziName;
    labels.value = project.labels;
    strokes.value = project.strokes;
    strokesBySource.value = project.strokesBySource ?? {};
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
