import { h, Fragment } from "preact";
import { useSignal } from "@preact/signals";
import * as state from "../state";
import { saveProject, loadProject } from "../state";

interface Props {
  onLoad: (url: string) => Promise<void>;
}

export function DataPanel({ onLoad }: Props) {
  const customUrl = useSignal(state.dziUrl.value);
  const loading = state.isLoadingImage;
  const error = state.loadError;
  const meta = state.dziMeta;
  const resolution = state.workLevelOffset;

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
    </section>
  );
}
