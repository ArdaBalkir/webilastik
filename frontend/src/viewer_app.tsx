import { h } from "preact";
import { useEffect, useRef } from "preact/hooks";
import { useSignal } from "@preact/signals";
import { DziViewer } from "./dzi_viewer";
import { SegmentationOverlay } from "./seg_overlay";
import { AtlasOverlay, type AtlasDisplayMode } from "./atlas_overlay";
import { ApiClient } from "./api";
import type { OverlayBlendMode, SourceEntry } from "./types";

/**
 * ViewerApp — clean overlay viewer
 * ----------------------------------
 * URL: ?mode=viewer&workdir=<data-proxy-dir>&token=<bearer>&server=<api-url>
 *      &registration=<webwarp-json-url>&atlas=<atlas-name>
 *
 * Shows source images from workdir/zipped_images/ with their matching
 * prediction overlay from workdir/segmentations/ — no annotation tools,
 * no training, no dzsave needed. Predictions can be loaded from either a
 * prediction DZIP or a browser-readable flat image.
 */

function readParam(key: string) {
  return new URLSearchParams(window.location.search).get(key) ?? "";
}

const BLEND_MODES: Array<{ value: OverlayBlendMode; label: string }> = [
  { value: "normal", label: "Normal" },
  { value: "multiply", label: "Multiply" },
  { value: "screen", label: "Screen" },
  { value: "overlay", label: "Overlay" },
  { value: "difference", label: "Difference" },
];

export function findSegmentationSource(
  sourceName: string,
  candidates: SourceEntry[],
): SourceEntry | null {
  const stem = (name: string) => name.replace(/\.[^.]+$/, "").toLowerCase();
  const sequence = (name: string): number | null => {
    const match = name.match(/(?:^|_)s(\d+)(?!\d)/i);
    return match ? Number.parseInt(match[1], 10) : null;
  };
  const usable = candidates.filter(({ name }) =>
    /\.(?:d?zip|png|jpe?g|webp|gif|bmp|avif)$/i.test(name),
  );
  const sourceStem = stem(sourceName);
  const sourceSequence = sequence(sourceName);
  return candidates.find(({ name }) => name === sourceName)
    ?? usable.find(({ name }) => stem(name) === sourceStem)
    ?? usable.find(({ name }) => sourceSequence !== null && sequence(name) === sourceSequence)
    ?? null;
}

export function ViewerApp() {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef    = useRef<DziViewer | null>(null);
  const segRef       = useRef<SegmentationOverlay | null>(null);
  const atlasRef     = useRef<AtlasOverlay | null>(null);
  const segmentationListRef = useRef<Promise<SourceEntry[]> | null>(null);

  const token     = useSignal(readParam("token"));
  const serverUrl = useSignal(readParam("server") || (import.meta.env.VITE_COMPUTE_SERVER_URL ?? "http://localhost:8000"));
  const workdir   = useSignal(readParam("workdir"));
  const registrationUrl = useSignal(
    readParam("registration") || readParam("registration_url"),
  );
  const atlasName = useSignal(readParam("atlas") || readParam("atlas_name"));

  const sources      = useSignal<SourceEntry[]>([]);
  const loadingList  = useSignal(false);
  const listError    = useSignal("");

  const activeSource  = useSignal<SourceEntry | null>(null);
  const loadingImage  = useSignal(false);
  const imageError    = useSignal("");

  const segVisible = useSignal(true);
  const segOpacity = useSignal(0.5);
  const segBlendMode = useSignal<OverlayBlendMode>("normal");
  const segStatus  = useSignal<"none" | "loading" | "ready" | "error">("none");
  const segError   = useSignal("");

  const atlasVisible = useSignal(true);
  const atlasOpacity = useSignal(0.5);
  const atlasDisplayMode = useSignal<AtlasDisplayMode>("fill");
  const atlasOutlineColor = useSignal("#0000ff");
  const atlasStatus = useSignal<"idle" | "loading" | "ready" | "error">("idle");
  const atlasProgress = useSignal("");
  const atlasError = useSignal("");
  const loadedAtlasName = useSignal("");

  // Init viewer + overlay once on mount
  useEffect(() => {
    const container = containerRef.current!;
    const viewer = new DziViewer(container);
    const seg    = new SegmentationOverlay(container, viewer);
    const atlas  = new AtlasOverlay(container, viewer);
    viewerRef.current = viewer;
    segRef.current    = seg;
    atlasRef.current  = atlas;
    return () => { atlas.destroy(); seg.destroy(); viewer.destroy(); };
  }, []);

  // Keep segmentation visibility / opacity in sync
  useEffect(() => segVisible.subscribe((v) => segRef.current?.setVisible(v)), []);
  useEffect(() => segOpacity.subscribe((o) => segRef.current?.setOpacity(o)), []);
  useEffect(() => segBlendMode.subscribe((mode) => segRef.current?.setBlendMode(mode)), []);
  useEffect(() => atlasVisible.subscribe((v) => atlasRef.current?.setVisible(v)), []);
  useEffect(() => atlasOpacity.subscribe((o) => atlasRef.current?.setOpacity(o)), []);
  useEffect(() => atlasDisplayMode.subscribe((mode) => atlasRef.current?.setDisplayMode(mode)), []);
  useEffect(() => atlasOutlineColor.subscribe((color) => atlasRef.current?.setOutlineColor(color)), []);

  // Load source list on mount (or when workdir changes)
  useEffect(() => {
    const dir = workdir.value;
    if (!dir) return;
    loadingList.value = true;
    listError.value   = "";
    const api = new ApiClient(serverUrl.value, token.value);
    const base = dir.replace(/\/$/, "");
    api.listSources(base + "/zipped_images/")
      .then((list) => { sources.value = list; })
      .catch((e)   => { listError.value = String(e); })
      .finally(()  => { loadingList.value = false; });
    segmentationListRef.current = api.listObjects(base + "/segmentations/", "");
  }, [workdir.value]);

  async function selectSource(src: SourceEntry) {
    activeSource.value  = src;
    imageError.value    = "";
    loadingImage.value  = true;
    segStatus.value     = "none";
    segRef.current?.clear();
    atlasRef.current?.clearSection();
    atlasError.value = "";

    try {
      const headers: Record<string, string> = token.value
        ? { Authorization: `Bearer ${token.value}` } : {};
      await viewerRef.current!.loadDzip(src.object_url, undefined, headers);
    } catch (e) {
      imageError.value   = String(e);
      loadingImage.value = false;
      return;
    }
    loadingImage.value = false;

    if (atlasStatus.value === "ready") displayAtlasForSource(src);

    const candidates = await segmentationListRef.current?.catch(() => []) ?? [];
    const match = findSegmentationSource(src.name, candidates);
    if (!match) {
      segStatus.value = "error";
      segError.value = `No matching segmentation for ${src.name}`;
      return;
    }
    segStatus.value = "loading";
    segError.value  = "";
    try {
      const headers: Record<string, string> = token.value
        ? { Authorization: `Bearer ${token.value}` } : {};
      await segRef.current!.load(match.object_url, headers);
      segStatus.value = "ready";
    } catch (e) {
      segStatus.value = "error";
      segError.value  = `Segmentation not found: ${e}`;
    }
  }

  function displayAtlasForSource(src: SourceEntry) {
    const candidates = [src.name, viewerRef.current?.dziName]
      .filter((name): name is string => !!name);
    let lastError: unknown;
    for (const name of new Set(candidates)) {
      try {
        atlasRef.current!.selectSource(name);
        atlasError.value = "";
        return;
      } catch (error) {
        lastError = error;
      }
    }
    atlasError.value = lastError instanceof Error
      ? lastError.message : String(lastError);
  }

  async function loadAtlasOverlay() {
    if (!registrationUrl.value || !atlasName.value || !atlasRef.current) return;
    atlasStatus.value = "loading";
    atlasProgress.value = "Loading registration…";
    atlasError.value = "";
    loadedAtlasName.value = "";
    try {
      const headers: Record<string, string> = token.value
        ? { Authorization: `Bearer ${token.value}` } : {};
      loadedAtlasName.value = await atlasRef.current.load(
        registrationUrl.value,
        atlasName.value,
        headers,
        (message) => { atlasProgress.value = message; },
      );
      atlasStatus.value = "ready";
      atlasProgress.value = "";
      if (activeSource.value) displayAtlasForSource(activeSource.value);
    } catch (error) {
      atlasStatus.value = "error";
      atlasProgress.value = "";
      atlasError.value = error instanceof Error ? error.message : String(error);
    }
  }

  const baseName = (url: string) => url.split("/").pop() ?? url;

  return (
    <div class="app-root">
      {/* ── Toolbar ─────────────────────────────────────────────────── */}
      <div class="viewer-toolbar">
        <span class="app-title" style="padding:0;border:none;margin:0">
          Webilastik Overlay Viewer
        </span>

        {/* Source picker */}
        <div class="viewer-toolbar-group">
          <label class="viewer-toolbar-label">Image</label>
          {loadingList.value && <span class="muted">loading…</span>}
          {listError.value && <span class="error">{listError.value}</span>}
          {!loadingList.value && sources.value.length === 0 && !listError.value && (
            <span class="muted">No images — check ?workdir=</span>
          )}
          {sources.value.length > 0 && (
            <select
              class="viewer-select"
              value={activeSource.value?.object_url ?? ""}
              onChange={(e) => {
                const url = (e.target as HTMLSelectElement).value;
                const src = sources.value.find((s) => s.object_url === url);
                if (src) selectSource(src);
              }}
            >
              <option value="">— pick an image —</option>
              {sources.value.map((s) => (
                <option key={s.object_url} value={s.object_url}>
                  {s.name}
                </option>
              ))}
            </select>
          )}
          {loadingImage.value && <span class="muted">loading image…</span>}
          {imageError.value  && <span class="error">{imageError.value}</span>}
        </div>

        {/* Segmentation overlay controls */}
        {activeSource.value && (
          <div class="viewer-toolbar-group">
            <label class="viewer-toolbar-label">Segmentation</label>
            {segStatus.value === "loading" && <span class="muted">loading…</span>}
            {segStatus.value === "error"   && <span class="error">{segError.value}</span>}
            {segStatus.value === "ready"   && (
              <>
                <label class="viewer-toolbar-check">
                  <input
                    type="checkbox"
                    checked={segVisible.value}
                    onChange={(e) =>
                      (segVisible.value = (e.target as HTMLInputElement).checked)
                    }
                  />
                  Show overlay
                </label>
                <label class="viewer-toolbar-check">
                  Opacity
                  <input
                    type="range"
                    min={0} max={100}
                    value={Math.round(segOpacity.value * 100)}
                    onInput={(e) =>
                      (segOpacity.value =
                        parseInt((e.target as HTMLInputElement).value) / 100)
                    }
                    style="width:80px"
                  />
                  <span class="muted">{Math.round(segOpacity.value * 100)}%</span>
                </label>
                <label class="viewer-toolbar-check">
                  Blend
                  <select
                    class="viewer-select"
                    value={segBlendMode.value}
                    onChange={(event) =>
                      (segBlendMode.value =
                        (event.target as HTMLSelectElement).value as OverlayBlendMode)
                    }
                  >
                    {BLEND_MODES.map((mode) => (
                      <option value={mode.value}>{mode.label}</option>
                    ))}
                  </select>
                </label>
              </>
            )}
          </div>
        )}

        {/* Atlas cut overlay controls */}
        <div class="viewer-toolbar-group">
          <label class="viewer-toolbar-label">Atlas</label>
          <button
            class="btn-sm"
            onClick={loadAtlasOverlay}
            disabled={
              atlasStatus.value === "loading" ||
              !registrationUrl.value ||
              !atlasName.value
            }
            title={
              !registrationUrl.value || !atlasName.value
                ? "Provide ?registration= and ?atlas= in the URL"
                : "Load the registered atlas cut"
            }
          >
            {atlasStatus.value === "loading" ? "Loading atlas…" : "Load atlas"}
          </button>
          {(!registrationUrl.value || !atlasName.value) && (
            <span class="muted">needs ?registration= and ?atlas=</span>
          )}
          {atlasStatus.value === "loading" && (
            <span class="muted">{atlasProgress.value}</span>
          )}
          {atlasError.value && <span class="error">{atlasError.value}</span>}
          {atlasStatus.value === "ready" && (
            <>
              <span class="muted">{loadedAtlasName.value}</span>
              <label class="viewer-toolbar-check">
                View
                <select
                  class="viewer-select"
                  value={atlasDisplayMode.value}
                  onChange={(event) =>
                    (atlasDisplayMode.value =
                      (event.target as HTMLSelectElement).value as AtlasDisplayMode)
                  }
                >
                  <option value="fill">Filled</option>
                  <option value="outline">Outline</option>
                </select>
              </label>
              <label class="viewer-toolbar-check">
                <input
                  type="checkbox"
                  checked={atlasVisible.value}
                  onChange={(event) =>
                    (atlasVisible.value =
                      (event.target as HTMLInputElement).checked)
                  }
                />
                Show atlas
              </label>
              {atlasDisplayMode.value === "outline" && (
                <label class="viewer-toolbar-check">
                  Color
                  <input
                    type="color"
                    value={atlasOutlineColor.value}
                    onInput={(event) =>
                      (atlasOutlineColor.value =
                        (event.target as HTMLInputElement).value)
                    }
                    title="Atlas outline color"
                  />
                </label>
              )}
              <label class="viewer-toolbar-check">
                Opacity
                <input
                  type="range"
                  min={0} max={100}
                  value={Math.round(atlasOpacity.value * 100)}
                  onInput={(event) =>
                    (atlasOpacity.value =
                      parseInt((event.target as HTMLInputElement).value) / 100)
                  }
                  style="width:80px"
                />
                <span class="muted">{Math.round(atlasOpacity.value * 100)}%</span>
              </label>
            </>
          )}
        </div>
      </div>

      {/* ── Canvas area ──────────────────────────────────────────────── */}
      <div class="viewer-area viewer-area-full" ref={containerRef} />
    </div>
  );
}
