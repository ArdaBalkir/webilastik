import { h, Fragment } from "preact";
import { useSignal } from "@preact/signals";
import * as state from "../state";
import { saveProject, loadProject } from "../state";
import { ApiClient, featureConfigToFilters } from "../api";
import type { ExportRequest } from "../types";
import { DataProxyDialog } from "./DataProxyDialog";

interface Props {
  onLoad: (url: string) => Promise<void>;
}

export function DataPanel({ onLoad }: Props) {
  const customUrl = useSignal(state.dziUrl.value);
  const loading = state.isLoadingImage;
  const error = state.loadError;
  const meta = state.dziMeta;
  const resolution = state.workLevelOffset;

  // Export state
  const outputUrl = useSignal("");
  const showDialog = useSignal(false);
  const uploading = useSignal(false);
  const uploadMsg = useSignal("");

  function normalizeUrl(raw: string): string {
    // Auto-fix common EBRAINS data-proxy URL mistake: missing /v1/buckets/
    // Wrong: https://data-proxy.ebrains.eu/api/my-bucket/path/file.dzip
    // Right: https://data-proxy.ebrains.eu/api/v1/buckets/my-bucket/path/file.dzip
    return raw.replace(
      /^(https?:\/\/data-proxy\.ebrains\.eu\/api\/)(?!v1\/buckets\/)(.+)$/,
      "$1v1/buckets/$2",
    );
  }

  function loadUrl(url: string) {
    const normalized = normalizeUrl(url.trim());
    customUrl.value = normalized;
    state.dziUrl.value = normalized;
    onLoad(normalized);
  }

  function handleFileInput(e: Event) {
    const input = e.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    loadUrl(url);
  }

  function handleProjectLoad(e: Event) {
    const input = e.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    loadProject(file).then(() => onLoad(state.dziUrl.value));
  }

  function buildExportReq(destUrl?: string): ExportRequest | null {
    const cid = state.classifierId.value;
    const wl = state.workLevel.value;
    const m = state.dziMeta.value;
    if (!cid || wl === null || !m) return null;
    const fc = state.featureConfig.value;
    return {
      classifier_id: cid,
      dzip_url: state.dziUrl.value,
      dzi_name: state.dziName.value,
      level: wl,
      features: { filters: featureConfigToFilters(fc), scales: fc.scales },
      ...(destUrl ? { output_url: destUrl } : {}),
    };
  }

  async function handleDownload() {
    const req = buildExportReq();
    if (!req) {
      uploadMsg.value = "Train a classifier first.";
      return;
    }
    uploadMsg.value = "Building DZIP…";
    try {
      await new ApiClient(
        state.serverUrl.value,
        state.bearerToken.value,
      ).exportZipDownload(req);
      uploadMsg.value = "Downloaded!";
    } catch (e) {
      uploadMsg.value = `Error: ${e}`;
    }
  }

  async function handleUpload() {
    const dest = outputUrl.value.trim();
    if (!dest) {
      uploadMsg.value = "Set a destination URL or browse first.";
      return;
    }
    const req = buildExportReq(dest);
    if (!req) {
      uploadMsg.value = "Train a classifier first.";
      return;
    }
    uploading.value = true;
    uploadMsg.value = "Uploading…";
    try {
      const res = await new ApiClient(
        state.serverUrl.value,
        state.bearerToken.value,
      ).exportZipUpload(req);
      uploadMsg.value = `✓ ${res.url} (${(res.size / 1024 / 1024).toFixed(1)} MB)`;
    } catch (e) {
      uploadMsg.value = `Error: ${e}`;
    }
    uploading.value = false;
  }

  return (
    <section class="panel">
      <h2>Image</h2>

      <div class="row">
        <input
          class="input-url"
          type="text"
          placeholder="DZIP URL…"
          value={customUrl.value}
          onInput={(e: Event) =>
            (customUrl.value = (e.target as HTMLInputElement).value)
          }
          onKeyDown={(e: KeyboardEvent) =>
            e.key === "Enter" && loadUrl(customUrl.value)
          }
        />
        <button
          class="btn"
          onClick={() => loadUrl(customUrl.value)}
          disabled={loading.value}
        >
          Open
        </button>
      </div>

      <div class="row">
        <label class="btn-file">
          Local file…{" "}
          <input type="file" accept=".dzip,.zip" onChange={handleFileInput} />
        </label>
      </div>

      {loading.value && <p class="status">Loading…</p>}
      {error.value && <p class="error">{error.value}</p>}

      {meta.value && (
        <div class="meta-info">
          <span>
            {Math.round(meta.value.width / Math.pow(2, resolution.value))} ×{" "}
            {Math.round(meta.value.height / Math.pow(2, resolution.value))} px
            {resolution.value > 0 && (
              <span class="meta-full-res">
                {" "}
                (full: {meta.value.width} × {meta.value.height})
              </span>
            )}
          </span>
          <span>
            Working level {meta.value.maxLevel - resolution.value} of{" "}
            {meta.value.maxLevel} · tile {meta.value.tileSize}px
          </span>
          <label class="row">
            Resolution:
            <select
              value={resolution.value}
              onChange={(e) =>
                (resolution.value = parseInt(
                  (e.target as HTMLSelectElement).value,
                ))
              }
            >
              <option value={0}>100% (full res)</option>
              <option value={1}>50%</option>
              <option value={2}>25%</option>
            </select>
          </label>
        </div>
      )}

      <div class="row divider-row">
        <button class="btn-sm" onClick={saveProject}>
          Save project
        </button>
        <label class="btn-sm btn-file">
          Load project…
          <input type="file" accept=".json" onChange={handleProjectLoad} />
        </label>
      </div>

      <div class="row">
        <label>Compute server:</label>
        <input
          class="input-url"
          type="text"
          value={state.serverUrl.value}
          onInput={(e) =>
            (state.serverUrl.value = (e.target as HTMLInputElement).value)
          }
        />
      </div>

      <div class="row">
        <label>Token:</label>
        <input
          class="input-url"
          type="password"
          placeholder="Bearer token"
          value={state.bearerToken.value}
          onInput={(e) =>
            (state.bearerToken.value = (e.target as HTMLInputElement).value)
          }
        />
      </div>

      {/* ── Export ────────────────────────────────────────────────────────── */}
      <div class="row divider-row">
        <label>Export predictions (DZIP)</label>
      </div>
      <div class="row">
        <input
          class="input-url"
          type="text"
          value={outputUrl.value}
          placeholder="data-proxy URL or blank → download"
          onInput={(e: Event) =>
            (outputUrl.value = (e.target as HTMLInputElement).value)
          }
        />
        <button
          class="btn-sm"
          title="Browse EBRAINS buckets"
          onClick={() => (showDialog.value = true)}
        >
          …
        </button>
      </div>
      <div class="row">
        <button
          class="btn-sm"
          onClick={handleDownload}
          disabled={state.trainingStatus.value !== "ready"}
        >
          ⬇ Download
        </button>
        <button
          class="btn-sm"
          onClick={handleUpload}
          disabled={state.trainingStatus.value !== "ready" || uploading.value}
        >
          {uploading.value ? "Uploading…" : "⬆ Upload to EBRAINS"}
        </button>
      </div>
      {uploadMsg.value && <p class="status">{uploadMsg.value}</p>}

      {showDialog.value && (
        <DataProxyDialog
          token={state.bearerToken.value}
          onSelect={(url) => {
            outputUrl.value = url;
          }}
          onClose={() => (showDialog.value = false)}
        />
      )}
    </section>
  );
}
