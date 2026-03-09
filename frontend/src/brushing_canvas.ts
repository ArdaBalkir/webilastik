import type { DziViewer } from "./dzi_viewer";
import type { Stroke, Label } from "./types";

export interface BrushEvent {
  stroke: Stroke;
}

/**
 * Transparent canvas overlay that sits on top of the DziViewer canvas.
 * Handles brush input and renders all stored strokes.
 *
 * Coordinate convention: strokes are stored in *full-resolution* image pixels,
 * matching the DziViewer's canvasToImage / imageToCanvas transforms.
 */
export class BrushingCanvas {
  readonly canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private viewer: DziViewer;

  private strokes: Stroke[] = [];
  private labels: Label[] = [];

  // Active brush state
  private activePoints: Array<[number, number]> = [];
  private activeLabelId = 1;
  private brushSize = 3; // radius in full-res image pixels
  private enabled = false;

  // Callbacks
  onStrokeFinished?: (stroke: Stroke) => void;

  constructor(container: HTMLElement, viewer: DziViewer) {
    this.viewer = viewer;
    this.canvas = document.createElement("canvas");
    this.canvas.style.cssText =
      "position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;";
    container.appendChild(this.canvas);
    this.ctx = this.canvas.getContext("2d")!;

    new ResizeObserver(() => this.handleResize()).observe(container);
    this.handleResize();

    // Redraw whenever the viewer pans/zooms — chain onto any existing callback
    const prevCallback = viewer.onViewChanged;
    viewer.onViewChanged = () => {
      prevCallback?.();
      this.redraw();
    };
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  setEnabled(enabled: boolean) {
    this.enabled = enabled;
    this.canvas.style.pointerEvents = enabled ? "auto" : "none";
    this.canvas.style.cursor = enabled ? "crosshair" : "default";
    if (!enabled) this.detachMouseEvents();
    else this.attachMouseEvents();
  }

  setBrushSize(radiusPx: number) {
    this.brushSize = radiusPx;
  }

  setActiveLabel(labelId: number) {
    this.activeLabelId = labelId;
  }

  setLabels(labels: Label[]) {
    this.labels = labels;
    this.redraw();
  }

  /** Replace all stored strokes (e.g. after project load). */
  setStrokes(strokes: Stroke[]) {
    this.strokes = strokes;
    this.redraw();
  }

  getStrokes(): Stroke[] {
    return this.strokes;
  }

  clearStrokes() {
    this.strokes = [];
    this.redraw();
  }

  // ── Resize ──────────────────────────────────────────────────────────────────

  private handleResize() {
    const w = this.canvas.offsetWidth || 800;
    const h = this.canvas.offsetHeight || 600;
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
      this.redraw();
    }
  }

  // ── Rendering ───────────────────────────────────────────────────────────────

  redraw() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    // Draw committed strokes
    for (const stroke of this.strokes) {
      const label = this.labels.find((l) => l.id === stroke.labelId);
      const color = label?.color ?? "#ff0000";
      this.drawStroke(stroke.points, color, stroke.level);
    }

    // Draw the in-progress stroke
    if (this.activePoints.length > 0) {
      const label = this.labels.find((l) => l.id === this.activeLabelId);
      const color = label?.color ?? "#ff0000";
      this.drawStroke(this.activePoints, color, null);
    }
  }

  /**
   * Draw a stroke as filled circles at each point.
   * Points are in DZI level coords when level is provided, otherwise full-res.
   */
  private drawStroke(
    points: Array<[number, number]>,
    color: string,
    strokeLevel: number | null,
  ) {
    if (points.length === 0) return;
    const ctx = this.ctx;
    const meta = this.viewer.meta_;

    ctx.save();
    ctx.fillStyle = hexToRgba(color, 0.65);

    for (const [px, py] of points) {
      let ix = px;
      let iy = py;
      // Convert from stroke level to full-res if needed
      if (strokeLevel !== null && meta) {
        const scale = Math.pow(2, strokeLevel - meta.maxLevel);
        ix = px / scale;
        iy = py / scale;
      }
      const [cx, cy] = this.viewer.imageToCanvas(ix, iy);
      const r = Math.max(1, this.brushSize * this.viewer.viewZoom);
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  // ── Mouse events ─────────────────────────────────────────────────────────────

  private mouseMoveHandler: ((e: MouseEvent) => void) | null = null;
  private mouseUpHandler: ((e: MouseEvent) => void) | null = null;

  private attachMouseEvents() {
    this.canvas.addEventListener("mousedown", this.onMouseDown);
  }

  private detachMouseEvents() {
    this.canvas.removeEventListener("mousedown", this.onMouseDown);
  }

  private onMouseDown = (e: MouseEvent) => {
    if (e.button !== 0) return;
    e.stopPropagation();

    const rect = this.canvas.getBoundingClientRect();
    const meta = this.viewer.meta_;
    const level = meta?.maxLevel ?? 0;

    const addPoint = (clientX: number, clientY: number) => {
      const cx = clientX - rect.left;
      const cy = clientY - rect.top;
      const [ix, iy] = this.viewer.canvasToImage(cx, cy);
      if (!meta) return;
      const scale = Math.pow(2, level - meta.maxLevel);
      const lx = Math.round(ix * scale);
      const ly = Math.round(iy * scale);
      const last = this.activePoints.at(-1);
      if (last && last[0] === lx && last[1] === ly) return; // deduplicate
      // Interpolate between last point and current for smooth strokes
      if (last && this.activePoints.length > 0) {
        const [x0, y0] = last;
        const steps = Math.max(1, Math.round(Math.hypot(lx - x0, ly - y0)));
        for (let i = 1; i <= steps; i++) {
          this.activePoints.push([
            Math.round(x0 + ((lx - x0) * i) / steps),
            Math.round(y0 + ((ly - y0) * i) / steps),
          ]);
        }
      } else {
        this.activePoints.push([lx, ly]);
      }
      this.redraw();
    };

    addPoint(e.clientX, e.clientY);

    this.mouseMoveHandler = (ev: MouseEvent) =>
      addPoint(ev.clientX, ev.clientY);
    this.mouseUpHandler = () => {
      if (this.activePoints.length > 0) {
        const stroke: Stroke = {
          labelId: this.activeLabelId,
          level,
          points: [...this.activePoints],
        };
        this.strokes.push(stroke);
        this.onStrokeFinished?.(stroke);
        this.activePoints = [];
        this.redraw();
      }
      document.removeEventListener("mousemove", this.mouseMoveHandler!);
      document.removeEventListener("mouseup", this.mouseUpHandler!);
      this.mouseMoveHandler = null;
      this.mouseUpHandler = null;
    };

    document.addEventListener("mousemove", this.mouseMoveHandler);
    document.addEventListener("mouseup", this.mouseUpHandler);
  };
}

// ── Prediction overlay ───────────────────────────────────────────────────────
/**
 * Semi-transparent canvas overlay that shows prediction tile PNGs from the
 * compute server. Re-fetches tiles whenever the view changes.
 */
export class PredictionOverlay {
  readonly canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private viewer: DziViewer;

  private tileCache = new Map<string, HTMLImageElement | "loading" | "error">();
  private getTileUrl:
    | ((level: number, col: number, row: number) => string)
    | null = null;
  private visible = false;
  private opacity = 0.5;
  private dirty = false;
  private rafId = 0;
  private lockedLevel: number | null = null;

  constructor(container: HTMLElement, viewer: DziViewer) {
    this.viewer = viewer;
    this.canvas = document.createElement("canvas");
    this.canvas.style.cssText =
      "position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;";
    container.appendChild(this.canvas);
    this.ctx = this.canvas.getContext("2d")!;

    new ResizeObserver(() => this.handleResize()).observe(container);
    this.handleResize();

    // Chain onto viewer's view changed callback
    const prevCallback = viewer.onViewChanged;
    viewer.onViewChanged = () => {
      prevCallback?.();
      this.dirty = true;
    };

    this.startLoop();
  }

  setTileUrlFn(fn: (level: number, col: number, row: number) => string) {
    this.getTileUrl = fn;
    this.tileCache.clear();
    this.dirty = true;
  }

  /** Lock prediction rendering to a specific DZI level (the trained level). */
  setLockedLevel(level: number | null) {
    this.lockedLevel = level;
    this.tileCache.clear();
    this.dirty = true;
  }

  setVisible(v: boolean) {
    this.visible = v;
    this.dirty = true;
  }

  setOpacity(o: number) {
    this.opacity = o;
    this.dirty = true;
  }

  clearCache() {
    this.tileCache.clear();
    this.dirty = true;
  }

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
    if (!this.visible || !this.getTileUrl || !viewer.meta_) return;

    const meta = viewer.meta_;
    const level =
      this.lockedLevel !== null
        ? this.lockedLevel
        : Math.max(
            0,
            Math.min(
              meta.maxLevel,
              Math.round(meta.maxLevel + Math.log2(viewer.viewZoom)),
            ),
          );
    const scale = Math.pow(2, level - meta.maxLevel);
    const lw = Math.max(1, Math.ceil(meta.width * scale));
    const lh = Math.max(1, Math.ceil(meta.height * scale));
    const ts = meta.tileSize;

    const [ix0, iy0] = viewer.canvasToImage(0, 0);
    const [ix1, iy1] = viewer.canvasToImage(canvas.width, canvas.height);

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

    ctx.save();
    ctx.globalAlpha = this.opacity;

    for (let row = rowMin; row <= rowMax; row++) {
      for (let col = colMin; col <= colMax; col++) {
        const key = `${level}/${col}_${row}`;
        const cached = this.tileCache.get(key);

        if (cached instanceof HTMLImageElement) {
          const leftOl = col === 0 ? 0 : meta.overlap;
          const topOl = row === 0 ? 0 : meta.overlap;
          const renderW = Math.min(ts, lw - col * ts);
          const renderH = Math.min(ts, lh - row * ts);
          const imgX = (col * ts) / scale;
          const imgY = (row * ts) / scale;
          const [cx, cy] = viewer.imageToCanvas(imgX, imgY);
          ctx.drawImage(
            cached,
            leftOl,
            topOl,
            renderW,
            renderH,
            cx,
            cy,
            (renderW / scale) * viewer.viewZoom,
            (renderH / scale) * viewer.viewZoom,
          );
        } else if (!cached) {
          this.tileCache.set(key, "loading");
          const url = this.getTileUrl!(level, col, row);
          const img = new Image();
          img.onload = () => {
            this.tileCache.set(key, img);
            this.dirty = true;
          };
          img.onerror = () => this.tileCache.set(key, "error");
          img.src = url;
        }
      }
    }
    ctx.restore();
  }

  destroy() {
    cancelAnimationFrame(this.rafId);
  }
}

// ── Utility ──────────────────────────────────────────────────────────────────

function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}
