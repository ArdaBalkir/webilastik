import type {
  DziInfoResponse,
  TrainRequest,
  TrainMultiRequest,
  TrainResponse,
  ExportRequest,
  ExportStatus,
  SessionInfo,
  FeatureConfig,
  BucketListEntry,
  StorageApiResponse,
  SourceEntry,
  BatchExportRequest,
  BatchExportStatus,
} from "./types";

export class ApiClient {
  constructor(
    private readonly baseUrl: string,
    private token?: string,
  ) {}

  setToken(token: string) {
    this.token = token;
  }

  private headers(): HeadersInit {
    const h: Record<string, string> = { "Content-Type": "application/json" };
    if (this.token) h["Authorization"] = `Bearer ${this.token}`;
    return h;
  }

  private async request<T>(
    method: string,
    path: string,
    body?: unknown,
    params?: Record<string, string>,
  ): Promise<T> {
    const url = new URL(path, this.baseUrl);
    if (params) {
      for (const [k, v] of Object.entries(params)) url.searchParams.set(k, v);
    }
    const res = await fetch(url.toString(), {
      method,
      headers: this.headers(),
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
    if (!res.ok) {
      const text = await res.text().catch(() => res.statusText);
      throw new Error(`${method} ${path} → ${res.status}: ${text}`);
    }
    return res.json() as Promise<T>;
  }

  /** Fetch DZI metadata from the compute server (it reads the DZIP). */
  async getDziInfo(dzipUrl: string): Promise<DziInfoResponse> {
    return this.request<DziInfoResponse>("GET", "/dzi-info", undefined, {
      dzip_url: dzipUrl,
    });
  }

  /** Send annotations + feature config to train a GPU/CPU Random Forest. */
  async train(req: TrainRequest): Promise<TrainResponse> {
    return this.request<TrainResponse>("POST", "/train", req);
  }

  /** Train on strokes from multiple images in one call. */
  async trainMulti(req: TrainMultiRequest): Promise<TrainResponse> {
    return this.request<TrainResponse>("POST", "/train-multi", req);
  }

  /**
   * Build the URL for a prediction tile PNG.
   * The browser fetches this directly so Neuroglancer-style caching headers apply.
   */
  predictionTileUrl(params: {
    classifierId: string;
    level: number;
    col: number;
    row: number;
    dzipUrl: string;
    dziName: string;
    featureConfig: FeatureConfig;
  }): string {
    const { classifierId, level, col, row, dzipUrl, dziName, featureConfig } =
      params;
    const filters = featureConfigToFilters(featureConfig).join(",");
    const scales = featureConfig.scales.join(",");
    const u = new URL(
      `/predict/${classifierId}/${level}/${col}_${row}`,
      this.baseUrl,
    );
    u.searchParams.set("dzip_url", dzipUrl);
    u.searchParams.set("dzi_name", dziName);
    u.searchParams.set("filters", filters);
    u.searchParams.set("scales", scales);
    // img.src can't send headers — pass token as query param as fallback
    if (this.token) u.searchParams.set("token", this.token);
    return u.toString();
  }

  async startExport(req: ExportRequest): Promise<{ job_id: string }> {
    return this.request<{ job_id: string }>("POST", "/export", req);
  }

  async getExportStatus(jobId: string): Promise<ExportStatus> {
    return this.request<ExportStatus>("GET", `/export/${jobId}`);
  }

  /**
   * POST /export-zip — builds a DZIP from prediction tiles.
   * If output_url is set, uploads it there and returns JSON.
   * If omitted, the response is a binary ZIP blob — call exportZipDownload instead.
   */
  async exportZipUpload(
    req: ExportRequest,
  ): Promise<{ status: string; url: string; filename: string; size: number }> {
    return this.request("POST", "/export-zip", req);
  }

  /** POST /export-zip with no output_url — triggers a browser file download. */
  async exportZipDownload(req: ExportRequest): Promise<void> {
    const res = await fetch(`${this.baseUrl}/export-zip`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(this.token ? { Authorization: `Bearer ${this.token}` } : {}),
      },
      body: JSON.stringify(req),
    });
    if (!res.ok) {
      const detail = await res.text();
      throw new Error(`Export failed (${res.status}): ${detail}`);
    }
    const blob = await res.blob();
    const disposition = res.headers.get("Content-Disposition") ?? "";
    const match = disposition.match(/filename="([^"]+)"/);
    const filename = match?.[1] ?? "predictions.dzip";
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  async listSources(dirUrl: string): Promise<SourceEntry[]> {
    return this.request(
      "GET",
      `/list-sources?url=${encodeURIComponent(dirUrl)}`,
    );
  }

  async startBatchExport(req: BatchExportRequest): Promise<{ job_id: string }> {
    return this.request("POST", "/batch-export", req);
  }

  async getBatchExportStatus(jobId: string): Promise<BatchExportStatus> {
    return this.request("GET", `/batch-export/${jobId}`);
  }
}

export interface HpcJobRequest {
  annotations: object[]; // [{dzip_url, strokes:[{label, points}]}]
  t_source?: string;
  features?: object;
  level?: number;
  p_source: string;
  output_dir: string;
  partition?: string;
  cpus?: number;
  mem?: string;
  time_limit?: string;
  account?: string;
}

export interface HpcJobStatus {
  job_id: string;
  slurm_job_id: string;
  slurm_state: string;
  status: "pending" | "running" | "done" | "error" | "cancelled";
  p_source: string;
  output_dir: string;
  log_path: string;
  created_at: number;
}

// ── Session allocator client ─────────────────────────────────────────────────
export class SessionAllocatorClient {
  constructor(
    private readonly baseUrl: string,
    private token: string,
  ) {}

  private headers(): HeadersInit {
    return {
      "Content-Type": "application/json",
      Authorization: `Bearer ${this.token}`,
    };
  }

  async createSession(): Promise<SessionInfo> {
    const res = await fetch(`${this.baseUrl}/sessions`, {
      method: "POST",
      headers: this.headers(),
    });
    if (!res.ok) throw new Error(`Create session failed: ${res.status}`);
    return res.json();
  }

  async getSession(sessionId: string): Promise<SessionInfo> {
    const res = await fetch(`${this.baseUrl}/sessions/${sessionId}`, {
      headers: this.headers(),
    });
    if (!res.ok) throw new Error(`Get session failed: ${res.status}`);
    return res.json();
  }

  /** Poll until the session is running (url is set) or fails. */
  async waitForSession(
    sessionId: string,
    onStatus?: (s: SessionInfo) => void,
    pollIntervalMs = 3000,
    timeoutMs = 300000,
  ): Promise<SessionInfo> {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      const info = await this.getSession(sessionId);
      onStatus?.(info);
      if (info.status === "running" && info.url) return info;
      if (info.status === "error") throw new Error("Session allocation failed");
      await sleep(pollIntervalMs);
    }
    throw new Error("Timed out waiting for session");
  }

  async deleteSession(sessionId: string): Promise<void> {
    await fetch(`${this.baseUrl}/sessions/${sessionId}`, {
      method: "DELETE",
      headers: this.headers(),
    });
  }

  // ── Headless HPC jobs ──────────────────────────────────────────────────────

  async submitHeadlessJob(req: HpcJobRequest): Promise<HpcJobStatus> {
    const res = await fetch(`${this.baseUrl}/headless-jobs`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(req),
    });
    if (!res.ok) {
      const text = await res.text().catch(() => res.statusText);
      throw new Error(`Submit job failed (${res.status}): ${text}`);
    }
    return res.json();
  }

  async getHeadlessJob(jobId: string): Promise<HpcJobStatus> {
    const res = await fetch(`${this.baseUrl}/headless-jobs/${jobId}`, {
      headers: this.headers(),
    });
    if (!res.ok) throw new Error(`Get job failed: ${res.status}`);
    return res.json();
  }

  async getJobLog(jobId: string, tail = 80): Promise<string> {
    const res = await fetch(
      `${this.baseUrl}/headless-jobs/${jobId}/log?tail=${tail}`,
      { headers: this.headers() },
    );
    if (!res.ok) return "(could not fetch log)";
    const data = await res.json();
    return data.log ?? "(empty)";
  }

  async cancelHeadlessJob(jobId: string): Promise<void> {
    await fetch(`${this.baseUrl}/headless-jobs/${jobId}`, {
      method: "DELETE",
      headers: this.headers(),
    });
  }

  /** Returns true if the allocator is reachable. */
  async checkHealth(): Promise<boolean> {
    try {
      const res = await fetch(`${this.baseUrl}/health`, {
        signal: AbortSignal.timeout(4000),
      });
      return res.ok;
    } catch {
      return false;
    }
  }
}

// ── Helpers ──────────────────────────────────────────────────────────────────
export function featureConfigToFilters(fc: FeatureConfig): string[] {
  const map: Array<[keyof FeatureConfig, string]> = [
    ["gaussianSmoothing", "gaussianSmoothing"],
    ["laplacianOfGaussian", "laplacianOfGaussian"],
    ["gaussianGradientMagnitude", "gaussianGradientMagnitude"],
    ["differenceOfGaussians", "differenceOfGaussians"],
    ["structureTensorEigenvalues", "structureTensorEigenvalues"],
    ["hessianOfGaussianEigenvalues", "hessianOfGaussianEigenvalues"],
  ];
  return map
    .filter(([k]) => (fc as unknown as Record<string, unknown>)[k] === true)
    .map(([, v]) => v);
}

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

// ── EBRAINS data-proxy — called directly from the browser ────────────────────────

const DATA_PROXY_BASE = "https://data-proxy.ebrains.eu/api/v1";

async function dpFetch<T>(url: string, token: string): Promise<T> {
  const res = await fetch(url, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok)
    throw new Error(`${res.status}: ${await res.text().catch(() => "")}`);
  return res.json() as Promise<T>;
}

export const dataProxy = {
  listBuckets(token: string): Promise<BucketListEntry[]> {
    return dpFetch(`${DATA_PROXY_BASE}/buckets`, token);
  },

  listObjects(
    token: string,
    bucket: string,
    prefix = "",
  ): Promise<StorageApiResponse> {
    const u = new URL(
      `${DATA_PROXY_BASE}/buckets/${encodeURIComponent(bucket)}`,
    );
    u.searchParams.set("delimiter", "/");
    if (prefix) u.searchParams.set("prefix", prefix);
    return dpFetch(u.toString(), token);
  },
};
