// ── DZI metadata ─────────────────────────────────────────────────────────────
export interface DziMeta {
  width: number;
  height: number;
  tileSize: number;
  overlap: number;
  format: string; // "jpeg" | "png"
  maxLevel: number; // ceil(log2(max(width, height))) — full-resolution level index
}

// ── Project model (browser-owned, serialised to JSON) ────────────────────────
export interface Label {
  id: number; // 1-based
  name: string;
  color: string; // CSS hex, e.g. "#ff0000"
}

export interface Stroke {
  labelId: number;
  level: number; // DZI level whose coordinate space the points are in
  points: Array<[number, number]>; // [x, y] in level-scaled image pixels
}

export interface Project {
  dziUrl: string; // URL of the .dzip file
  dziName: string; // name of the .dzi entry inside the zip (no .dzi suffix)
  workLevel: number | null; // null = auto (maxLevel)
  labels: Label[];
  strokes: Stroke[];
}

// ── Feature configuration ────────────────────────────────────────────────────
export interface FeatureConfig {
  gaussianSmoothing: boolean;
  laplacianOfGaussian: boolean;
  gaussianGradientMagnitude: boolean;
  differenceOfGaussians: boolean;
  structureTensorEigenvalues: boolean;
  hessianOfGaussianEigenvalues: boolean;
  scales: number[];
}

export const DEFAULT_FEATURE_CONFIG: FeatureConfig = {
  gaussianSmoothing: true,
  laplacianOfGaussian: true,
  gaussianGradientMagnitude: true,
  differenceOfGaussians: true,
  structureTensorEigenvalues: true,
  hessianOfGaussianEigenvalues: true,
  scales: [0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0],
};

// ── API request / response shapes ────────────────────────────────────────────
export interface DziInfoResponse {
  name: string;
  width: number;
  height: number;
  tileSize: number;
  overlap: number;
  format: string;
  maxLevel: number;
}

export interface TrainRequest {
  dzip_url: string;
  dzi_name: string;
  level: number;
  strokes: Array<{
    label: number;
    points: Array<[number, number]>;
  }>;
  features: {
    filters: string[];
    scales: number[];
  };
}

export interface TrainMultiRequest {
  /** Same structure as the exported annotations JSON — works for 1 or N images. */
  annotations: Array<{
    dzip_url: string;
    level?: number; // DZI level strokes were drawn at; omit = max_level (full res)
    strokes: Array<{
      label: number;
      points: Array<[number, number]>;
    }>;
  }>;
  features: {
    filters: string[];
    scales: number[];
  };
}

export interface TrainResponse {
  classifier_id: string;
  num_classes: number;
}

export interface ExportRequest {
  classifier_id: string;
  dzip_url: string;
  dzi_name: string;
  level: number;
  features: {
    filters: string[];
    scales: number[];
  };
  output_url?: string; // if omitted → browser download; if set → PUT to that URL
}

export interface ExportStatus {
  status: "pending" | "running" | "done" | "error";
  progress?: number; // 0..1
  error?: string;
  url?: string;
}

// ── Source listing ────────────────────────────────────────────────────────────
export interface SourceEntry {
  name: string;
  object_url: string;
  bytes: number | null;
}

// ── Batch export ──────────────────────────────────────────────────────────────
export interface BatchExportRequest {
  classifier_id: string;
  p_source: string;
  output_dir: string;
  features: { filters: string[]; scales: number[] };
}

export interface BatchExportStatus {
  status: "pending" | "running" | "done" | "error";
  progress: number;
  total: number;
  done: number;
  failed: Array<{ name: string; error: string }>;
  current?: string;
  error?: string;
}

// ── HPC job history record (stored in localStorage) ──────────────────────────
export interface HpcJobRecord {
  job_id: string;
  slurm_job_id: string;
  slurm_state: string;
  status: "pending" | "running" | "done" | "error" | "cancelled";
  p_source: string;
  output_dir: string;
  log_path: string;
  created_at: number; // unix ms
  annotated_images: number;
}

// ── Headless run ──────────────────────────────────────────────────────────────
export interface HeadlessRequest {
  t_source: string;
  annotations: Array<{
    dzip_url: string;
    strokes: Array<{ label: number; points: Array<[number, number]> }>;
  }>;
  features: { filters: string[]; scales: number[] };
  level: number;
  p_source: string;
  output_dir: string;
}
// ── EBRAINS data-proxy ───────────────────────────────────────────────────────────
export interface BucketListEntry {
  name: string;
  role: "administrator" | "editor" | "viewer" | null;
  is_public: boolean;
}
export interface StorageObject {
  name: string;
  bytes: number;
  last_modified?: string | null;
}
export interface StorageDir {
  subdir: string;
  bytes: number | null;
  last_modified?: string | null;
}
export interface StorageApiResponse {
  objects: (StorageObject | StorageDir)[];
  container: string;
  prefix: string | null;
  marker: string | null;
}
// ── Auth ─────────────────────────────────────────────────────────────────────
export interface SessionInfo {
  session_id: string;
  status: "pending" | "running" | "done" | "error";
  url?: string; // compute server URL once running
}
