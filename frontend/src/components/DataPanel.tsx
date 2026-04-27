import { h } from "preact";
import { useSignal } from "@preact/signals";
import * as state from "../state";
import { loadProject, pinSources, unpinSource } from "../state";
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
  // Tracks which source_urls are ticked in the picker
  const checkedUrls = useSignal<Set<string>>(new Set());

  function normalizeUrl(raw: string): string {
    return raw.replace(
      /^(https?:\/\/data-proxy\.ebrains\.eu\/api\/)(?!v1\/buckets\/)(.+)$/,
      "$1v1/buckets/$2",
    );
  }

  function loadUrl(url: string) {
    const normalized = normalizeUrl(url.trim());
    customUrl.value = normalized;
    // Save current strokes before switching, restore any saved strokes for new URL
    state.switchTrainingSource(normalized);
    onLoad(normalized);
  }

  async function openBrowse() {
    showBrowse.value = true;
    checkedUrls.value = new Set();
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

  function toggleCheck(url: string) {
    const next = new Set(checkedUrls.value);
    if (next.has(url)) next.delete(url); else next.add(url);
    checkedUrls.value = next;
  }

  // Cloud project picker
  const showProjectPicker = useSignal(false);
  const projectEntries = useSignal<SourceEntry[]>([]);
  const projectPickerLoading = useSignal(false);
  const projectPickerError = useSignal("");

  async function openProjectPicker() {
    showProjectPicker.value = true;
    projectPickerError.value = "";
    projectEntries.value = [];
    const currentUrl = state.dziUrl.value;
    const match = currentUrl.match(/\/v1\/buckets\/([^/]+)\//);
    if (!match) {
      projectPickerError.value = "Open a data-proxy image first so the bucket is known.";
      return;
    }
    const bucket = match[1];
    const dir = `https://data-proxy.ebrains.eu/api/v1/buckets/${bucket}/ilastikProjectSaves/`;
    projectPickerLoading.value = true;
    try {
      projectEntries.value = await new ApiClient(
        state.serverUrl.value, state.bearerToken.value,
      ).listObjects(dir, ".json");
    } catch (e) {
      projectPickerError.value = String(e);
    }
    projectPickerLoading.value = false;
  }

  async function loadCloudProject(objectUrl: string) {
    showProjectPicker.value = false;
    try {
      const headers: Record<string, string> = state.bearerToken.value
        ? { Authorization: `Bearer ${state.bearerToken.value}` } : {};
      const res = await fetch(objectUrl, { headers });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const text = await res.text();
      const file = new File([text], "project.json", { type: "application/json" });
      await loadProject(file);
      onLoad(state.dziUrl.value);
    } catch (e) {
      state.loadError.value = `Cloud load failed: ${e}`;
    }
  }

  function confirmSelection() {
    const selected = browseSources.value.filter((s) =>
      checkedUrls.value.has(s.object_url),
    );
    if (selected.length > 0) pinSources(selected);
    showBrowse.value = false;
    checkedUrls.value = new Set();
  }

  return (
    <section class="panel">
      <h2>Image</h2>

      {/* Quick-switch strip — always visible when images are pinned */}
      {state.pinnedSources.value.length > 0 && (
        <div class="quick-switch">
          <span class="quick-switch-label">Quick switch</span>
          <ul class="quick-switch-list">
            {state.pinnedSources.value.map((s) => (
              <li
                key={s.object_url}
                class={`quick-switch-item${
                  state.dziUrl.value === s.object_url ? " active" : ""
                }`}
              >
                <button
                  class="quick-switch-btn"
                  title={s.object_url}
                  onClick={() => loadUrl(s.object_url)}
                >
                  {s.name}
                  {Object.keys(state.strokesBySource.value).includes(s.object_url) &&
                    (state.strokesBySource.value[s.object_url]?.length ?? 0) > 0 && (
                    <span class="stroke-dot" title="Has annotations" />
                  )}
                </button>
                <button
                  class="btn-icon quick-switch-remove"
                  title="Remove from list"
                  onClick={(e) => { e.stopPropagation(); unpinSource(s.object_url); }}
                >×</button>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Primary action: Browse data-proxy */}
      <button
        class="btn btn-train"
        style={{ marginBottom: 4 }}
        onClick={openBrowse}
        disabled={browseLoading.value}
      >
        {browseLoading.value ? "Loading…" : "Select training images"}
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
            {browseSources.value.map((s) => {
              const checked = checkedUrls.value.has(s.object_url);
              return (
                <li
                  key={s.object_url}
                  class={`source-item${state.dziUrl.value === s.object_url ? " active" : ""}${
                    checked ? " checked" : ""
                  }`}
                  onClick={() => toggleCheck(s.object_url)}
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => toggleCheck(s.object_url)}
                    onClick={(e) => e.stopPropagation()}
                  />
                  <span class="source-name">{s.name}</span>
                </li>
              );
            })}
          </ul>
          <div class="row" style={{ justifyContent: "space-between", marginTop: 2 }}>
            <button
              class="btn btn-train"
              style={{ flex: 1 }}
              disabled={checkedUrls.value.size === 0}
              onClick={confirmSelection}
            >
              Add {checkedUrls.value.size > 0 ? checkedUrls.value.size : ""} to quick-switch
            </button>
          </div>
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
        <button class="btn-sm" onClick={openProjectPicker}>Load cloud project…</button>
      </div>

      {/* Cloud project picker */}
      {showProjectPicker.value && (
        <div class="source-picker">
          <div class="row" style={{ justifyContent: "space-between" }}>
            <span class="hint">ilastikProjectSaves/</span>
            <button class="btn-icon" onClick={() => (showProjectPicker.value = false)}>×</button>
          </div>
          {projectPickerLoading.value && <p class="status">Loading…</p>}
          {projectPickerError.value && <p class="status error">{projectPickerError.value}</p>}
          <ul class="source-list">
            {projectEntries.value.map((s) => (
              <li key={s.object_url} class="source-item" onClick={() => loadCloudProject(s.object_url)}>
                <span class="source-name">{s.name}</span>
              </li>
            ))}
            {!projectPickerLoading.value && projectEntries.value.length === 0 && !projectPickerError.value && (
              <li class="source-item" style={{ color: "#666" }}>No saved projects found</li>
            )}
          </ul>
        </div>
      )}

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
