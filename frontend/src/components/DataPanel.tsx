import { h } from "preact";
import { useSignal } from "@preact/signals";
import * as state from "../state";
import { saveProject, loadProject } from "../state";
import { ApiClient } from "../api";
import type { SourceEntry } from "../types";

interface Props {
  onLoad: (url: string) => Promise<void>;
}

export function DataPanel({ onLoad }: Props) {
  const customUrl = useSignal(state.dziUrl.value);
  const loading = state.isLoadingImage;
  const error = state.loadError;
  const meta = state.dziMeta;
  const resolution = state.workLevelOffset;
  const showAdvanced = useSignal(false);

  // Inline source browser
  const showBrowse = useSignal(false);
  const browseSources = useSignal<SourceEntry[]>([]);
  const browseLoading = useSignal(false);
  const browseError = useSignal("");

  function normalizeUrl(raw: string): string {
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
    loadUrl(URL.createObjectURL(file));
  }

  function handleProjectLoad(e: Event) {
    const input = e.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    loadProject(file).then(() => onLoad(state.dziUrl.value));
  }

  async function openBrowse() {
    showBrowse.value = true;
    const dir = state.sourceDir.value;
    if (!dir) { browseError.value = "Set ?workdir= in the URL first."; return; }
    browseLoading.value = true;
    browseError.value = "";
    try {
      browseSources.value = await new ApiClient(
        state.serverUrl.value, state.bearerToken.value,
      ).listSources(dir);
    } catch (e) {
      browseError.value = String(e);
    }
    browseLoading.value = false;
  }

  return (
    <section class="panel">
      <h2>Image</h2>

      {/* Primary action: Browse data-proxy */}
      <button
        class="btn btn-train"
        style={{ marginBottom: 4 }}
        onClick={openBrowse}
        disabled={browseLoading.value}
      >
        {browseLoading.value ? "Loading…" : "Browse data-proxy"}
      </button>

      {/* Inline source picker */}
      {showBrowse.value && (
        <div class="source-picker">
          <div class="row" style={{ justifyContent: "space-between" }}>
            <span class="hint" style={{ wordBreak: "break-all" }}>
              {state.sourceDir.value || "(no workdir set)"}
            </span>
            <button class="btn-icon" onClick={() => (showBrowse.value = false)}>×</button>
          </div>
          {browseLoading.value && <p class="status">Loading…</p>}
          {browseError.value && <p class="status error">{browseError.value}</p>}
          <ul class="source-list">
            {browseSources.value.map((s) => (
              <li
                key={s.object_url}
                class={`source-item${state.dziUrl.value === s.object_url ? " active" : ""}`}
                onClick={() => { showBrowse.value = false; loadUrl(s.object_url); }}
              >
                <span class="source-name">{s.name}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

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
        <button class="btn-sm" onClick={saveProject}>Save project</button>
        <label class="btn-sm btn-file">
          Load project…
          <input type="file" accept=".json" onChange={handleProjectLoad} />
        </label>
      </div>

      {/* Advanced: server, token, direct URL, local file */}
      <details class="adv-details">
        <summary class="adv-summary">Advanced</summary>
        <div class="adv-body">
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
            <label class="btn-sm btn-file">
              Local file…{" "}
              <input type="file" accept=".dzip,.zip" onChange={handleFileInput} />
            </label>
          </div>

          <div class="row">
            <label class="hint">Server:</label>
            <input
              class="input-url"
              type="text"
              value={state.serverUrl.value}
              onInput={(e) => (state.serverUrl.value = (e.target as HTMLInputElement).value)}
            />
          </div>

          <div class="row">
            <label class="hint">Token:</label>
            <input
              class="input-url"
              type="password"
              placeholder="Bearer token"
              value={state.bearerToken.value}
              onInput={(e) => (state.bearerToken.value = (e.target as HTMLInputElement).value)}
            />
          </div>
        </div>
      </details>
    </section>
  );
}
