import { h } from "preact";
import { useEffect, useRef } from "preact/hooks";
import { useSignal } from "@preact/signals";
import { DziViewer } from "./dzi_viewer";
import { SegmentationOverlay } from "./seg_overlay";
import { ApiClient } from "./api";
import type { SourceEntry } from "./types";

/**
 * ViewerApp — clean overlay viewer
 * ----------------------------------
 * URL: ?mode=viewer&workdir=<data-proxy-dir>&token=<bearer>&server=<api-url>
 *
 * Shows source images from workdir/zipped_images/ with their matching
 * prediction overlay from workdir/segmentations/ — no annotation tools,
 * no training, no dzsave needed. Predictions can be loaded from either a
 * prediction DZIP or a browser-readable flat image.
 */

function readParam(key: string) {
  return new URLSearchParams(window.location.search).get(key) ?? "";
}

const OVERLAY_EXTENSIONS = new Set([
  ".dzip", ".zip", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".avif",
]);

function splitExtension(name: string): { stem: string; extension: string } {
  const dot = name.lastIndexOf(".");
  if (dot <= 0) return { stem: name, extension: "" };
  return {
    stem: name.slice(0, dot),
    extension: name.slice(dot).toLowerCase(),
  };
}

/** Pick the stored overlay that belongs to a source image. */
export function findSegmentationSource(
  sourceName: string,
  candidates: SourceEntry[],
): SourceEntry | null {
  // Preserve the original convention first: segmentations/foo.dzip matches
  // zipped_images/foo.dzip exactly.
  const exact = candidates.find((candidate) => candidate.name === sourceName);
  if (exact) return exact;

  const sourceStem = splitExtension(sourceName).stem.toLowerCase();
  const usable = candidates.filter((candidate) =>
    OVERLAY_EXTENSIONS.has(splitExtension(candidate.name).extension),
  );

  // Also accept a different container/format with the same basename, such as
  // foo.png for foo.dzip.
  const sameStem = usable.find(
    (candidate) => splitExtension(candidate.name).stem.toLowerCase() === sourceStem,
  );
  if (sameStem) return sameStem;

  // Basenames can differ while retaining the image sequence id (`_s0000`).
  const sequenceId = sourceStem.match(/(?:^|_)(s\d{4})(?:_|$)/)?.[1];
  if (sequenceId) {
    const sameSequence = usable.find((candidate) =>
      splitExtension(candidate.name).stem.toLowerCase()
        .match(/(?:^|_)(s\d{4})(?:_|$)/)?.[1] === sequenceId,
    );
    if (sameSequence) return sameSequence;
  }

  // Single-export output commonly uses this suffix.
  return usable.find(
    (candidate) =>
      splitExtension(candidate.name).stem.toLowerCase() === `${sourceStem}_predictions`,
  ) ?? null;
}

export function ViewerApp() {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef    = useRef<DziViewer | null>(null);
  const segRef       = useRef<SegmentationOverlay | null>(null);
  const segmentationListRef = useRef<Promise<SourceEntry[]> | null>(null);

  const token     = useSignal(readParam("token"));
  const serverUrl = useSignal(readParam("server") || (import.meta.env.VITE_COMPUTE_SERVER_URL ?? "http://localhost:8000"));
  const workdir   = useSignal(readParam("workdir"));

  const sources      = useSignal<SourceEntry[]>([]);
  const segmentations = useSignal<SourceEntry[]>([]);
  const segmentationListError = useSignal("");
  const loadingList  = useSignal(false);
  const listError    = useSignal("");

  const activeSource  = useSignal<SourceEntry | null>(null);
  const loadingImage  = useSignal(false);
  const imageError    = useSignal("");

  const segVisible = useSignal(true);
  const segOpacity = useSignal(0.5);
  const segStatus  = useSignal<"none" | "loading" | "ready" | "error">("none");
  const segError   = useSignal("");

  // Init viewer + overlay once on mount
  useEffect(() => {
    const container = containerRef.current!;
    const viewer = new DziViewer(container);
    const seg    = new SegmentationOverlay(container, viewer);
    viewerRef.current = viewer;
    segRef.current    = seg;
    return () => { viewer.destroy(); seg.destroy(); };
  }, []);

  // Keep segmentation visibility / opacity in sync
  useEffect(() => segVisible.subscribe((v) => segRef.current?.setVisible(v)), []);
  useEffect(() => segOpacity.subscribe((o) => segRef.current?.setOpacity(o)), []);

  // Load source list on mount (or when workdir changes)
  useEffect(() => {
    const dir = workdir.value;
    if (!dir) return;
    loadingList.value = true;
    listError.value   = "";
    segmentationListError.value = "";
    const api = new ApiClient(serverUrl.value, token.value);
    const base = dir.replace(/\/$/, "");
    api.listSources(base + "/zipped_images/")
      .then((list) => { sources.value = list; })
      .catch((e)   => { listError.value = String(e); })
      .finally(()  => { loadingList.value = false; });
    // An empty extension deliberately lists every object. Filtering and
    // basename matching happen locally so raster overlays can be discovered.
    const segmentationRequest = api.listObjects(base + "/segmentations/", "")
      .then((list) => { segmentations.value = list; })
      .catch((e) => {
        segmentations.value = [];
        segmentationListError.value = String(e);
      });
    segmentationListRef.current = segmentationRequest.then(() => segmentations.value);
  }, [workdir.value]);

  async function selectSource(src: SourceEntry) {
    activeSource.value  = src;
    imageError.value    = "";
    loadingImage.value  = true;
    segStatus.value     = "none";
    segRef.current?.clear();

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

    // Auto-load a matching segmentation. Exact object names remain preferred,
    // with same-basename raster/DZIP files accepted as a fallback.
    const segBase = workdir.value.replace(/\/$/, "") + "/segmentations/";
    const availableSegmentations = segmentationListRef.current
      ? await segmentationListRef.current
      : segmentations.value;
    const match = findSegmentationSource(src.name, availableSegmentations);
    const segUrl = match?.object_url ?? segBase + src.name;
    segStatus.value = "loading";
    segError.value  = "";
    try {
      const headers: Record<string, string> = token.value
        ? { Authorization: `Bearer ${token.value}` } : {};
      await segRef.current!.load(segUrl, headers);
      segStatus.value = "ready";
    } catch (e) {
      segStatus.value = "error";
      const listingDetail = segmentationListError.value
        ? ` (overlay listing also failed: ${segmentationListError.value})`
        : "";
      segError.value  = `Segmentation not found: ${e}${listingDetail}`;
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
              </>
            )}
          </div>
        )}
      </div>

      {/* ── Canvas area ──────────────────────────────────────────────── */}
      <div class="viewer-area viewer-area-full" ref={containerRef} />
    </div>
  );
}
