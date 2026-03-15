/**
 * SourceBrowser — training image selector
 *
 * Shows all .dzip files found at `t_source` (from URL param or typed).
 * User ticks which ones to annotate.  Clicking an item loads it into the viewer.
 * Strokes are preserved per-image via state.strokesBySource.
 */
import { h } from "preact";
import { useSignal } from "@preact/signals";
import { ApiClient } from "../api";
import * as state from "../state";
import type { SourceEntry } from "../types";

interface Props {
  onLoad: (url: string) => Promise<void>;
}

export function SourceBrowser({ onLoad }: Props) {
  const open = useSignal(false);
  const dirUrl = useSignal(state.tSourceUrl.value);
  const loading = state.trainingSourcesLoading;
  const error = state.trainingSourcesError;
  const sources = state.trainingSources;
  const selected = state.selectedTrainingSources;
  const activeUrl = state.dziUrl;

  async function fetchSources(url: string) {
    if (!url.trim()) return;
    loading.value = true;
    error.value = "";
    try {
      const client = new ApiClient(
        state.serverUrl.value,
        state.bearerToken.value,
      );
      const results = await client.listSources(url.trim());
      sources.value = results;
      // Auto-select all
      selected.value = new Set(results.map((r) => r.object_url));
    } catch (e) {
      error.value = String(e);
    } finally {
      loading.value = false;
    }
  }

  function toggleSource(entry: SourceEntry) {
    const next = new Set(selected.value);
    if (next.has(entry.object_url)) {
      next.delete(entry.object_url);
    } else {
      next.add(entry.object_url);
    }
    selected.value = next;
  }

  async function activate(entry: SourceEntry) {
    // Save current strokes before switching
    state.switchTrainingSource(entry.object_url);
    await onLoad(entry.object_url);
  }

  function fmtBytes(b: number | null): string {
    if (b == null) return "";
    if (b < 1024 * 1024) return `${(b / 1024).toFixed(0)} KB`;
    return `${(b / 1024 / 1024).toFixed(1)} MB`;
  }

  return (
    <section class="panel collapsible">
      <button
        class="collapsible-header"
        onClick={() => {
          open.value = !open.value;
          if (open.value && !sources.value.length && dirUrl.value)
            fetchSources(dirUrl.value);
        }}
      >
        <span>Training Sources</span>
        <span class="chevron">{open.value ? "▲" : "▼"}</span>
      </button>

      {open.value && (
        <div class="collapsible-body">
          <div class="row">
            <input
              class="input-url"
              value={dirUrl.value}
              placeholder="data-proxy dir URL…"
              onInput={(e: Event) =>
                (dirUrl.value = (e.target as HTMLInputElement).value)
              }
            />
            <button
              class="btn-sm"
              disabled={loading.value}
              onClick={() => fetchSources(dirUrl.value)}
            >
              {loading.value ? "…" : "Load"}
            </button>
          </div>

          {error.value && <p class="status error">{error.value}</p>}

          {sources.value.length > 0 && (
            <>
              <div class="row" style="justify-content:space-between">
                <span class="hint">
                  {selected.value.size}/{sources.value.length} selected
                </span>
                <button
                  class="btn-sm"
                  onClick={() => {
                    if (selected.value.size === sources.value.length) {
                      selected.value = new Set();
                    } else {
                      selected.value = new Set(
                        sources.value.map((s) => s.object_url),
                      );
                    }
                  }}
                >
                  {selected.value.size === sources.value.length
                    ? "None"
                    : "All"}
                </button>
              </div>

              <ul class="source-list">
                {sources.value.map((entry) => {
                  const isActive = activeUrl.value === entry.object_url;
                  const isSel = selected.value.has(entry.object_url);
                  const hasStrokes =
                    (state.strokesBySource.value[entry.object_url]?.length ??
                      0) > 0 ||
                    (isActive && state.strokes.value.length > 0);
                  return (
                    <li
                      key={entry.object_url}
                      class={`source-item${isActive ? " active" : ""}`}
                      onClick={() => activate(entry)}
                    >
                      <input
                        type="checkbox"
                        checked={isSel}
                        onClick={(e: Event) => {
                          e.stopPropagation();
                          toggleSource(entry);
                        }}
                      />
                      <span class="source-name" title={entry.object_url}>
                        {entry.name}
                      </span>
                      <span class="source-meta">
                        {hasStrokes && (
                          <span class="dot-annotated" title="has annotations" />
                        )}
                        {fmtBytes(entry.bytes)}
                      </span>
                    </li>
                  );
                })}
              </ul>
            </>
          )}
        </div>
      )}
    </section>
  );
}
