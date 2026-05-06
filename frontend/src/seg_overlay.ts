import netunzip, { NetUnzipDirectory } from "./dzip_helper";
import type { DziViewer } from "./dzi_viewer";
import type { DziMeta } from "./types";

/**
 * SegmentationOverlay
 * -------------------
 * Loads a pre-computed prediction DZIP from data-proxy and renders it as a
 * transparent canvas layer on top of a DziViewer.
 *
 * Prediction DZIPs contain tiles at a single DZI level (the level the
 * classifier was run at — always max_level unless overridden).  We lock
 * rendering to that level and let the canvas scale tiles at lower zoom
 * levels, so the overlay is always visible regardless of zoom.
 */
export class SegmentationOverlay {
  readonly canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private viewer: DziViewer;

  private dzip: NetUnzipDirectory | null = null;
  private segName = "";      // DZIP entry prefix, e.g. "foo_predictions"
  private segLevel = 0;      // the single level that exists in the DZIP
  private segMeta: DziMeta | null = null;

  private tileCache = new Map<string, HTMLImageElement | "loading" | "error">();
  private blobUrls = new Map<string, string>();

  private visible = true;
  private opacity = 0.5;
  private dirty = false;
  private rafId = 0;

  constructor(container: HTMLElement, viewer: DziViewer) {
    this.viewer = viewer;
    this.canvas = document.createElement("canvas");
    this.canvas.style.cssText =
      "position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;";
    container.appendChild(this.canvas);
    this.ctx = this.canvas.getContext("2d")!;

    new ResizeObserver(() => this.handleResize()).observe(container);
    this.handleResize();

    // Chain onto viewer's view changed callback so we redraw on pan/zoom.
    const prev = viewer.onViewChanged;
    viewer.onViewChanged = () => {
      prev?.();
      this.dirty = true;
    };

    this.startLoop();
  }

  // ── Public API ───────────────────────────────────────────────────────────

  /**
   * Load a prediction DZIP.  Pass the object URL and optional bearer token.
   * Returns the segmentation name (usable for display).
   */
  async load(
    dzipUrl: string,
    extraHeaders?: Record<string, string>,
  ): Promise<string> {
    this.clear();
    this.dzip = await netunzip(dzipUrl, extraHeaders);

    // Find the .dzi file inside the archive to determine segName and level.
    const dziEntry = [...this.dzip.entries.keys()].find((k) => k.endsWith(".dzi"));
    if (!dziEntry) throw new Error("No .dzi found in segmentation DZIP");

    this.segName = dziEntry.slice(0, -4); // strip ".dzi"

    const xmlBytes = await this.dzip.get(this.dzip.entries.get(dziEntry)!);
    const xml = new TextDecoder().decode(xmlBytes);
    const { parseDzi } = await import("./dzi_viewer");
    this.segMeta = parseDzi(xml);
    this.segLevel = this.segMeta.maxLevel;

    this.dirty = true;
    return this.segName;
  }

  clear() {
    for (const u of this.blobUrls.values()) URL.revokeObjectURL(u);
    this.blobUrls.clear();
    this.tileCache.clear();
    this.dzip = null;
    this.segMeta = null;
    this.dirty = true;
  }

  setVisible(v: boolean) { this.visible = v; this.dirty = true; }
  setOpacity(o: number)  { this.opacity = o;  this.dirty = true; }

  destroy() {
    cancelAnimationFrame(this.rafId);
    this.clear();
  }

  // ── Internal ─────────────────────────────────────────────────────────────

  private handleResize() {
    const w = this.canvas.offsetWidth || 800;
    const h = this.canvas.offsetHeight || 600;
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
      this.dirty = true;
    }
  }

  private startLoop() {
    const frame = () => {
      this.rafId = requestAnimationFrame(frame);
      if (!this.dirty) return;
      this.dirty = false;
      this.render();
    };
    this.rafId = requestAnimationFrame(frame);
  }

  private render() {
    const { ctx, canvas, viewer } = this;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!this.visible || !this.dzip || !this.segMeta || !viewer.meta_) return;

    const segMeta = this.segMeta;
    const srcMeta = viewer.meta_;
    const ts = segMeta.tileSize;
    const lw = segMeta.width;
    const lh = segMeta.height;

    // Scale: how many source full-res pixels correspond to one seg pixel.
    // e.g. if seg is quarter-res (export level = maxLevel-2), segToSrc = 4.
    const segToSrc = srcMeta.width / segMeta.width;

    // Visible source full-res region.
    const [ix0, iy0] = viewer.canvasToImage(0, 0);
    const [ix1, iy1] = viewer.canvasToImage(canvas.width, canvas.height);

    // Convert source coords to seg pixel coords for tile index math.
    const colMin = Math.max(0, Math.floor(ix0 / (ts * segToSrc)));
    const colMax = Math.min(Math.ceil(lw / ts) - 1, Math.floor(ix1 / (ts * segToSrc)));
    const rowMin = Math.max(0, Math.floor(iy0 / (ts * segToSrc)));
    const rowMax = Math.min(Math.ceil(lh / ts) - 1, Math.floor(iy1 / (ts * segToSrc)));

    ctx.save();
    ctx.globalAlpha = this.opacity;
    ctx.imageSmoothingEnabled = false;

    for (let row = rowMin; row <= rowMax; row++) {
      for (let col = colMin; col <= colMax; col++) {
        this.drawTile(col, row, segToSrc, lw, lh);
      }
    }

    ctx.restore();
  }

  private drawTile(
    col: number,
    row: number,
    segToSrc: number,
    lw: number,
    lh: number,
  ) {
    const { ctx, viewer } = this;
    const meta = this.segMeta!;
    const level = this.segLevel;
    const key = `${level}/${col}_${row}`;
    const cached = this.tileCache.get(key);

    if (cached instanceof HTMLImageElement) {
      const ts = meta.tileSize;
      const ol = meta.overlap;
      const leftOl = col === 0 ? 0 : ol;
      const topOl  = row === 0 ? 0 : ol;
      const renderW = Math.min(ts, lw - col * ts);
      const renderH = Math.min(ts, lh - row * ts);
      // Map seg tile top-left to source full-res coords
      const imgX = col * ts * segToSrc;
      const imgY = row * ts * segToSrc;
      const imgW = renderW * segToSrc;
      const imgH = renderH * segToSrc;
      const [cx, cy] = viewer.imageToCanvas(imgX, imgY);
      ctx.drawImage(
        cached,
        leftOl, topOl, renderW, renderH,
        cx, cy, imgW * viewer.viewZoom, imgH * viewer.viewZoom,
      );
      return;
    }

    if (cached === "loading" || cached === "error") return;

    this.tileCache.set(key, "loading");
    const path = `${this.segName}_files/${level}/${col}_${row}.png`;
    const entry = this.dzip?.entries.get(path);
    if (!entry) { this.tileCache.set(key, "error"); return; }

    this.dzip!.get(entry)
      .then((bytes) => {
        const blobUrl = URL.createObjectURL(
          new Blob([bytes.buffer as ArrayBuffer], { type: "image/png" }),
        );
        this.blobUrls.set(key, blobUrl);
        const img = new Image();
        img.onload = () => { this.tileCache.set(key, img); this.dirty = true; };
        img.onerror = () => this.tileCache.set(key, "error");
        img.src = blobUrl;
      })
      .catch(() => this.tileCache.set(key, "error"));
  }
}
