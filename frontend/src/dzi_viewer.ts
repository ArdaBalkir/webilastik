import netunzip, { NetUnzipDirectory } from "./dzip_helper";
import type { DziMeta } from "./types";

type TileState = HTMLImageElement | "loading" | "error";

export class DziViewer {
  readonly canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;

  private dzip: NetUnzipDirectory | null = null;
  private meta: DziMeta | null = null;
  dziName = "";

  // View state: (panX, panY) = full-res image coord at canvas centre
  // zoom = canvas pixels per full-res image pixel
  private panX = 0;
  private panY = 0;
  private zoom = 0;
  private _workLevel: number | null = null;

  private tileCache = new Map<string, TileState>();
  private blobUrls = new Map<string, string>(); // kept to revoke on destroy
  private dirty = false;
  private rafId = 0;
  private hasAnimatedLoadingTiles = false;
  private prefersReducedMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)",
  ).matches;

  // Pan gesture
  private isPanning = false;
  private panStart = { cx: 0, cy: 0, panX: 0, panY: 0 };

  // Callbacks
  onViewChanged?: () => void;

  constructor(container: HTMLElement) {
    this.canvas = document.createElement("canvas");
    this.canvas.style.cssText =
      "position:absolute;top:0;left:0;width:100%;height:100%;display:block;";
    container.appendChild(this.canvas);
    this.ctx = this.canvas.getContext("2d")!;

    new ResizeObserver(() => this.handleResize()).observe(container);
    this.handleResize();
    this.attachPointerEvents();
    this.startLoop();
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  async loadDzip(
    url: string,
    dziName?: string,
    extraHeaders?: Record<string, string>,
  ): Promise<DziMeta> {
    this.dzip = await netunzip(url, extraHeaders);
    // Discover the .dzi file if dziName not provided
    if (!dziName) {
      const found = [...this.dzip.entries.keys()].find((k) =>
        k.endsWith(".dzi"),
      );
      if (!found) throw new Error("No .dzi file found in archive");
      dziName = found.slice(0, -4);
    }
    this.dziName = dziName;

    const entry = this.dzip.entries.get(`${dziName}.dzi`);
    if (!entry) throw new Error(`${dziName}.dzi not found in archive`);
    const xmlBytes = await this.dzip.get(entry);
    const xml = new TextDecoder().decode(xmlBytes);
    const meta = parseDzi(xml);
    this.meta = meta;

    // Centre the image and fit it
    this.panX = meta.width / 2;
    this.panY = meta.height / 2;
    this.zoom =
      Math.min(
        this.canvas.width / meta.width,
        this.canvas.height / meta.height,
      ) * 0.9;
    this.tileCache.clear();
    this.dirty = true;
    return meta;
  }

  /** Convert canvas pixel coordinates to full-res image coordinates. */
  canvasToImage(cx: number, cy: number): [number, number] {
    return [
      this.panX + (cx - this.canvas.width / 2) / this.zoom,
      this.panY + (cy - this.canvas.height / 2) / this.zoom,
    ];
  }

  /** Convert full-res image coordinates to canvas pixel coordinates. */
  imageToCanvas(ix: number, iy: number): [number, number] {
    return [
      (ix - this.panX) * this.zoom + this.canvas.width / 2,
      (iy - this.panY) * this.zoom + this.canvas.height / 2,
    ];
  }

  get viewZoom() {
    return this.zoom;
  }
  get meta_() {
    return this.meta;
  }

  /** Cap the viewer to this DZI level (null = uncapped, use zoom). */
  setWorkLevel(level: number | null) {
    this._workLevel = level;
    this.tileCache.clear();
    for (const u of this.blobUrls.values()) URL.revokeObjectURL(u);
    this.blobUrls.clear();
    this.dirty = true;
  }

  invalidate() {
    this.dirty = true;
  }

  destroy() {
    cancelAnimationFrame(this.rafId);
    for (const u of this.blobUrls.values()) URL.revokeObjectURL(u);
  }

  // ── Resize ──────────────────────────────────────────────────────────────────

  private handleResize() {
    const w = this.canvas.offsetWidth || 800;
    const h = this.canvas.offsetHeight || 600;
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
      this.dirty = true;
    }
  }

  // ── Render loop ─────────────────────────────────────────────────────────────

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
    const { ctx, canvas, meta, dzip } = this;
    this.hasAnimatedLoadingTiles = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    // Disable bilinear smoothing when zoomed in so individual pixels are crisp.
    // Keep it on when zoomed out so downscaled tiles look smooth.
    ctx.imageSmoothingEnabled = this.zoom < 1;

    if (!meta || !dzip) return;

    const level = this.bestLevel();
    const scale = Math.pow(2, level - meta.maxLevel); // fraction of full-res
    const lw = Math.max(1, Math.ceil(meta.width * scale));
    const lh = Math.max(1, Math.ceil(meta.height * scale));
    const ts = meta.tileSize;

    // Visible full-res area
    const [ix0, iy0] = this.canvasToImage(0, 0);
    const [ix1, iy1] = this.canvasToImage(canvas.width, canvas.height);

    // Convert to level coords and tile indices
    const colMin = Math.max(0, Math.floor((ix0 * scale) / ts));
    const colMax = Math.min(
      Math.ceil(lw / ts) - 1,
      Math.floor((ix1 * scale) / ts),
    );
    const rowMin = Math.max(0, Math.floor((iy0 * scale) / ts));
    const rowMax = Math.min(
      Math.ceil(lh / ts) - 1,
      Math.floor((iy1 * scale) / ts),
    );

    for (let row = rowMin; row <= rowMax; row++) {
      for (let col = colMin; col <= colMax; col++) {
        this.drawTile(level, col, row, scale, lw, lh);
      }
    }

    // Keep the canvas ticking only while a visible shimmer is in progress.
    if (this.hasAnimatedLoadingTiles) this.dirty = true;
  }

  private drawTile(
    level: number,
    col: number,
    row: number,
    scale: number,
    lw: number,
    lh: number,
  ) {
    const { ctx, meta } = this;
    if (!meta) return;
    const key = `${level}/${col}_${row}`;
    const cached = this.tileCache.get(key);

    const ts = meta.tileSize;
    const tlx = col * ts;
    const tly = row * ts;
    const renderW = Math.min(ts, lw - tlx);
    const renderH = Math.min(ts, lh - tly);
    const imgX = tlx / scale;
    const imgY = tly / scale;
    const [cx, cy] = this.imageToCanvas(imgX, imgY);
    const canvasW = (renderW / scale) * this.zoom;
    const canvasH = (renderH / scale) * this.zoom;

    if (cached instanceof HTMLImageElement) {
      const ol = meta.overlap;
      // How many overlap pixels are actually present on left/top edges
      const leftOl = col === 0 ? 0 : ol;
      const topOl = row === 0 ? 0 : ol;
      ctx.drawImage(
        cached,
        leftOl,
        topOl,
        renderW,
        renderH,
        cx,
        cy,
        canvasW,
        canvasH,
      );
      return;
    }

    if (cached === "loading") {
      this.drawLoadingTile(cx, cy, canvasW, canvasH);
      return;
    }
    if (cached === "error") return;

    // Kick off async load
    this.tileCache.set(key, "loading");
    this.drawLoadingTile(cx, cy, canvasW, canvasH);
    const path = `${this.dziName}_files/${level}/${col}_${row}.${meta.format}`;
    const entry = this.dzip?.entries.get(path);
    if (!entry) {
      this.tileCache.set(key, "error");
      return;
    }

    this.dzip!.get(entry)
      .then((bytes) => {
        const mime = meta.format === "png" ? "image/png" : "image/jpeg";
        const blobUrl = URL.createObjectURL(
          new Blob([bytes.buffer as ArrayBuffer], { type: mime }),
        );
        this.blobUrls.set(key, blobUrl);
        const img = new Image();
        img.onload = () => {
          this.tileCache.set(key, img);
          this.dirty = true;
        };
        img.onerror = () => this.tileCache.set(key, "error");
        img.src = blobUrl;
      })
      .catch(() => this.tileCache.set(key, "error"));
  }

  /** Draw a quiet skeleton in exactly the space occupied by a pending tile. */
  private drawLoadingTile(x: number, y: number, width: number, height: number) {
    const { ctx, canvas } = this;

    ctx.fillStyle = "#121817";
    ctx.fillRect(x, y, width, height);

    if (this.prefersReducedMotion) return;
    this.hasAnimatedLoadingTiles = true;

    // One shared sweep across the viewport keeps adjacent tiles feeling like
    // a single image placeholder rather than a flickering checkerboard.
    const phase = (performance.now() % 1500) / 1500;
    const bandCenter = -canvas.width * 0.25 + phase * canvas.width * 1.5;
    const bandWidth = Math.max(90, Math.min(220, canvas.width * 0.16));
    const shimmer = ctx.createLinearGradient(
      bandCenter - bandWidth,
      0,
      bandCenter + bandWidth,
      0,
    );
    shimmer.addColorStop(0, "rgba(61, 214, 192, 0)");
    shimmer.addColorStop(0.5, "rgba(61, 214, 192, 0.14)");
    shimmer.addColorStop(1, "rgba(61, 214, 192, 0)");
    ctx.fillStyle = shimmer;
    ctx.fillRect(x, y, width, height);
  }

  /** Choose the DZI level that most closely matches the current zoom. */
  private bestLevel(): number {
    if (!this.meta) return 0;
    // DZI level L: 1 level pixel = 1/scale full-res pixels, scale = 2^(L - maxLevel)
    // We want scale ≈ zoom → L = maxLevel + log2(zoom)
    const zoomLevel = this.meta.maxLevel + Math.log2(this.zoom);
    const cap = this._workLevel ?? this.meta.maxLevel;
    return Math.max(0, Math.min(cap, Math.round(zoomLevel)));
  }

  // ── Pointer events (pan + zoom) ─────────────────────────────────────────────

  private attachPointerEvents() {
    const c = this.canvas;

    c.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        const rect = c.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const [ix, iy] = this.canvasToImage(mx, my);
        const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
        this.zoom *= factor;
        // Keep the image point under the cursor fixed
        this.panX = ix - (mx - c.width / 2) / this.zoom;
        this.panY = iy - (my - c.height / 2) / this.zoom;
        this.dirty = true;
        this.onViewChanged?.();
      },
      { passive: false },
    );

    c.addEventListener("mousedown", (e) => {
      if (e.button !== 0) return;
      this.isPanning = true;
      this.panStart = {
        cx: e.clientX,
        cy: e.clientY,
        panX: this.panX,
        panY: this.panY,
      };
      c.style.cursor = "grabbing";
    });

    // Listen on document so drag works even when cursor leaves canvas
    document.addEventListener("mousemove", (e) => {
      if (!this.isPanning) return;
      this.panX =
        this.panStart.panX - (e.clientX - this.panStart.cx) / this.zoom;
      this.panY =
        this.panStart.panY - (e.clientY - this.panStart.cy) / this.zoom;
      this.dirty = true;
      this.onViewChanged?.();
    });

    document.addEventListener("mouseup", () => {
      if (!this.isPanning) return;
      this.isPanning = false;
      c.style.cursor = "default";
    });

    // Touch support: single-finger pan, two-finger pinch-zoom
    let lastTouches: TouchList | null = null;
    c.addEventListener(
      "touchstart",
      (e) => {
        lastTouches = e.touches;
      },
      { passive: true },
    );
    c.addEventListener(
      "touchmove",
      (e) => {
        e.preventDefault();
        if (!lastTouches) return;
        const cur = e.touches;
        if (cur.length === 1 && lastTouches.length === 1) {
          const dx = (cur[0].clientX - lastTouches[0].clientX) / this.zoom;
          const dy = (cur[0].clientY - lastTouches[0].clientY) / this.zoom;
          this.panX -= dx;
          this.panY -= dy;
          this.dirty = true;
          this.onViewChanged?.();
        } else if (cur.length === 2 && lastTouches.length === 2) {
          const d0 = dist(lastTouches[0], lastTouches[1]);
          const d1 = dist(cur[0], cur[1]);
          if (d0 > 0) {
            const mx =
              (cur[0].clientX + cur[1].clientX) / 2 -
              c.getBoundingClientRect().left;
            const my =
              (cur[0].clientY + cur[1].clientY) / 2 -
              c.getBoundingClientRect().top;
            const [ix, iy] = this.canvasToImage(mx, my);
            this.zoom *= d1 / d0;
            this.panX = ix - (mx - c.width / 2) / this.zoom;
            this.panY = iy - (my - c.height / 2) / this.zoom;
            this.dirty = true;
            this.onViewChanged?.();
          }
        }
        lastTouches = cur;
      },
      { passive: false },
    );
    c.addEventListener(
      "touchend",
      (e) => {
        lastTouches = e.touches.length ? e.touches : null;
      },
      { passive: true },
    );
  }
}

// ── DZI XML parser ───────────────────────────────────────────────────────────

export function parseDzi(xml: string): DziMeta {
  const doc = new DOMParser().parseFromString(xml, "text/xml");
  const img = doc.querySelector("Image")!;
  const size = doc.querySelector("Size")!;
  const width = parseInt(size.getAttribute("Width")!);
  const height = parseInt(size.getAttribute("Height")!);
  const tileSize = parseInt(img.getAttribute("TileSize")!);
  const overlap = parseInt(img.getAttribute("Overlap") ?? "0");
  const format = (img.getAttribute("Format") ?? "jpeg").toLowerCase();
  const maxLevel = Math.ceil(Math.log2(Math.max(width, height)));
  return { width, height, tileSize, overlap, format, maxLevel };
}

// ── Utility ──────────────────────────────────────────────────────────────────

function dist(a: Touch, b: Touch) {
  return Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
}
