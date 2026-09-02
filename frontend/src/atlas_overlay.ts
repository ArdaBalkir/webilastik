import type { DziViewer } from "./dzi_viewer";
import type { OverlayBlendMode } from "./types";

interface AtlasLabel {
  rgb?: string;
  name?: string;
}

interface AtlasDescriptor {
  name: string;
  encoding: number;
  xdim: number;
  ydim: number;
  zdim: number;
  labels: AtlasLabel[];
}

interface AtlasVolume extends AtlasDescriptor {
  data: Uint8Array | Uint16Array;
}

type Marker = { x: number; y: number; nx: number; ny: number };

interface RegistrationSection {
  filename?: string;
  name?: string;
  nr?: number;
  width?: number;
  height?: number;
  ouv?: number[];
  anchoring?: number[];
  markers?: Array<Marker | [number, number, number, number]>;
}

interface RegistrationFile {
  sections?: RegistrationSection[];
  slices?: RegistrationSection[];
}

type AtlasWorkerMessage =
  | { type: "progress"; message: string }
  | { type: "ready"; atlas: AtlasDescriptor; buffer: ArrayBuffer }
  | { type: "error"; message: string };

type Point = [number, number];
type Triangle = [number, number, number];
export type AtlasDisplayMode = "fill" | "outline";

/** A WebWarp registration plane rendered over the active DZI image. */
export class AtlasOverlay {
  readonly canvas: HTMLCanvasElement;

  private readonly ctx: CanvasRenderingContext2D;
  private readonly viewer: DziViewer;
  private readonly resizeObserver: ResizeObserver;
  private atlas: AtlasVolume | null = null;
  private registration: RegistrationFile | null = null;
  private renderedCut: AtlasCut | null = null;
  private sliceCanvas: HTMLCanvasElement | null = null;
  private visible = true;
  private opacity = 0.5;
  private displayMode: AtlasDisplayMode = "fill";
  private smoothEdges = true;
  private dirty = false;
  private rafId = 0;

  constructor(container: HTMLElement, viewer: DziViewer) {
    this.viewer = viewer;
    this.canvas = document.createElement("canvas");
    this.canvas.style.cssText =
      "position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;";
    container.appendChild(this.canvas);
    this.ctx = this.canvas.getContext("2d")!;

    this.resizeObserver = new ResizeObserver(() => this.handleResize());
    this.resizeObserver.observe(container);
    this.handleResize();

    const previous = viewer.onViewChanged;
    viewer.onViewChanged = () => {
      previous?.();
      this.dirty = true;
    };
    this.startLoop();
  }

  async load(
    registrationUrl: string,
    atlasName: string,
    extraHeaders?: Record<string, string>,
    onProgress?: (message: string) => void,
  ): Promise<string> {
    this.clear();
    const [registration, atlas] = await Promise.all([
      fetchRegistration(registrationUrl, extraHeaders),
      loadAtlas(atlasName, onProgress),
    ]);
    this.registration = registration;
    this.atlas = atlas;
    this.dirty = true;
    return atlas.name;
  }

  /** Select and cut the registration entry matching an image filename. */
  selectSource(sourceName: string): string {
    this.renderedCut = null;
    this.sliceCanvas = null;
    if (!this.registration || !this.atlas) {
      this.dirty = true;
      throw new Error("Atlas has not been loaded");
    }
    const section = findRegistrationSection(sourceName, this.registration);
    if (!section) {
      this.dirty = true;
      throw new Error(`No atlas registration found for ${sourceName}`);
    }
    const plane = section.ouv ?? section.anchoring;
    if (!isPlane(plane)) {
      this.dirty = true;
      throw new Error(`Registration for ${sourceName} has no valid anchoring plane`);
    }

    const cut = cutAtlas(this.atlas, plane);
    const markers = normalizeMarkers(section.markers);
    this.renderedCut = markers.length > 0
      ? warpCut(cut, markers, section.width, section.height)
      : cut;
    this.rebuildSliceCanvas();
    this.dirty = true;
    return section.filename ?? section.name ?? sourceName;
  }

  clearSection() {
    this.renderedCut = null;
    this.sliceCanvas = null;
    this.dirty = true;
  }

  clear() {
    this.atlas = null;
    this.registration = null;
    this.clearSection();
  }

  setVisible(visible: boolean) {
    this.visible = visible;
    this.dirty = true;
  }

  setOpacity(opacity: number) {
    this.opacity = opacity;
    this.dirty = true;
  }

  setDisplayMode(mode: AtlasDisplayMode) {
    this.displayMode = mode;
    this.rebuildSliceCanvas();
  }

  setSmoothEdges(smooth: boolean) {
    this.smoothEdges = smooth;
    this.dirty = true;
  }

  setBlendMode(mode: OverlayBlendMode) {
    this.canvas.style.mixBlendMode = mode;
  }

  destroy() {
    cancelAnimationFrame(this.rafId);
    this.resizeObserver.disconnect();
    this.clear();
    this.canvas.remove();
  }

  private handleResize() {
    const width = this.canvas.offsetWidth || 800;
    const height = this.canvas.offsetHeight || 600;
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
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
    const { ctx, canvas, viewer, sliceCanvas } = this;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!this.visible || !sliceCanvas || !viewer.meta_) return;

    const [canvasX, canvasY] = viewer.imageToCanvas(0, 0);
    ctx.save();
    ctx.globalAlpha = this.opacity;
    ctx.imageSmoothingEnabled = this.smoothEdges;
    if (this.smoothEdges) ctx.imageSmoothingQuality = "high";
    ctx.drawImage(
      sliceCanvas,
      canvasX,
      canvasY,
      viewer.meta_.width * viewer.viewZoom,
      viewer.meta_.height * viewer.viewZoom,
    );
    ctx.restore();
  }

  private rebuildSliceCanvas() {
    this.sliceCanvas = this.renderedCut && this.atlas
      ? colorCut(this.renderedCut, this.atlas.labels, this.displayMode)
      : null;
    this.dirty = true;
  }
}

async function fetchRegistration(
  sourceUrl: string,
  extraHeaders?: Record<string, string>,
): Promise<RegistrationFile> {
  let url = new URL(sourceUrl, window.location.href).toString();
  let headers = extraHeaders;
  if (extraHeaders?.Authorization && url.includes("data-proxy.ebrains.eu")) {
    const proxyUrl = new URL(url);
    proxyUrl.searchParams.set("redirect", "false");
    const proxyResponse = await fetch(proxyUrl, { headers: extraHeaders });
    if (!proxyResponse.ok) {
      throw new Error(`Registration request failed (${proxyResponse.status})`);
    }
    const location = await proxyResponse.json();
    url = location.url ?? location.URL ?? String(Object.values(location)[0]);
    headers = undefined;
  }

  const response = await fetch(url, { headers });
  if (!response.ok) {
    throw new Error(`Registration request failed (${response.status})`);
  }
  const registration = (await response.json()) as RegistrationFile;
  if (!Array.isArray(registration.sections) && !Array.isArray(registration.slices)) {
    throw new Error("Registration file contains neither sections nor slices");
  }
  return registration;
}

function loadAtlas(
  atlasName: string,
  onProgress?: (message: string) => void,
): Promise<AtlasVolume> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL("./atlas_worker.ts", import.meta.url), {
      type: "module",
    });
    worker.onmessage = (event: MessageEvent<AtlasWorkerMessage>) => {
      const message = event.data;
      if (message.type === "progress") {
        onProgress?.(message.message);
        return;
      }
      worker.terminate();
      if (message.type === "error") {
        reject(new Error(message.message));
        return;
      }
      const data = message.atlas.encoding === 1
        ? new Uint8Array(message.buffer)
        : new Uint16Array(message.buffer);
      resolve({ ...message.atlas, data });
    };
    worker.onerror = (event) => {
      worker.terminate();
      reject(new Error(event.message || "Atlas worker failed"));
    };
    worker.postMessage({ atlasName });
  });
}

export function findRegistrationSection(
  sourceName: string,
  registration: RegistrationFile,
): RegistrationSection | null {
  const sections = registration.sections ?? registration.slices ?? [];
  const sourceKey = filenameKey(sourceName);
  const sourceSequence = sequenceNumber(sourceName);
  return sections.find((section) => {
    const name = section.filename ?? section.name ?? "";
    return name === sourceName;
  }) ?? sections.find((section) => {
    const name = section.filename ?? section.name ?? "";
    return !!name && filenameKey(name) === sourceKey;
  }) ?? sections.find((section) => {
    const name = section.filename ?? section.name ?? "";
    const sequence = section.nr ?? sequenceNumber(name);
    return sourceSequence !== null && sequence === sourceSequence;
  }) ?? null;
}

function filenameKey(value: string) {
  let name = value.split(/[\\/]/).pop()?.toLowerCase() ?? value.toLowerCase();
  name = name.replace(/\.(?:d?zip)$/i, "");
  return name.replace(/\.(?:png|jpe?g|tiff?|webp|bmp|avif)$/i, "");
}

function sequenceNumber(value: string): number | null {
  const match = value.match(/(?:^|_)s(\d+)(?!\d)/i);
  return match ? Number.parseInt(match[1], 10) : null;
}

function isPlane(value: number[] | undefined): value is number[] {
  return Array.isArray(value) && value.length >= 9 &&
    value.slice(0, 9).every(Number.isFinite);
}

function normalizeMarkers(
  markers: RegistrationSection["markers"],
): Marker[] {
  if (!markers) return [];
  return markers.map((marker) => Array.isArray(marker)
    ? { x: marker[0], y: marker[1], nx: marker[2], ny: marker[3] }
    : marker
  ).filter((marker) =>
    [marker.x, marker.y, marker.nx, marker.ny].every(Number.isFinite),
  );
}

interface AtlasCut {
  ids: Uint16Array;
  width: number;
  height: number;
}

function cutAtlas(atlas: AtlasVolume, plane: number[]): AtlasCut {
  const [ox, oy, oz, ux, uy, uz, vx, vy, vz] = plane;
  const width = Math.max(1, Math.round(Math.hypot(ux, uy, uz)));
  const height = Math.max(1, Math.round(Math.hypot(vx, vy, vz)));
  const ids = new Uint16Array(width * height);
  const zStride = atlas.xdim * atlas.ydim;

  for (let y = 0; y < height; y++) {
    const rowX = ox + vx * y / height;
    const rowY = oy + vy * y / height;
    const rowZ = oz + vz * y / height;
    for (let x = 0; x < width; x++) {
      const atlasX = Math.round(rowX + ux * x / width);
      const atlasY = Math.round(rowY + uy * x / width);
      const atlasZ = Math.round(rowZ + uz * x / width);
      if (
        atlasX >= 0 && atlasX < atlas.xdim &&
        atlasY >= 0 && atlasY < atlas.ydim &&
        atlasZ >= 0 && atlasZ < atlas.zdim
      ) {
        ids[x + y * width] =
          atlas.data[atlasX + atlasY * atlas.xdim + atlasZ * zStride];
      }
    }
  }
  return { ids, width, height };
}

function colorCut(
  cut: AtlasCut,
  labels: AtlasLabel[],
  mode: AtlasDisplayMode,
): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = cut.width;
  canvas.height = cut.height;
  const ctx = canvas.getContext("2d")!;
  const image = ctx.createImageData(cut.width, cut.height);
  for (let index = 0; index < cut.ids.length; index++) {
    const labelId = cut.ids[index];
    if (mode === "fill" || isBoundary(cut, index, labelId)) {
      writeLabelColor(image.data, index, labelId, labels);
    }
  }
  ctx.putImageData(image, 0, 0);
  return canvas;
}

function isBoundary(cut: AtlasCut, index: number, labelId: number) {
  if (labelId === 0) return false;
  const x = index % cut.width;
  const y = Math.floor(index / cut.width);
  return x === 0 || y === 0 || x === cut.width - 1 || y === cut.height - 1 ||
    cut.ids[index - 1] !== labelId || cut.ids[index + 1] !== labelId ||
    cut.ids[index - cut.width] !== labelId ||
    cut.ids[index + cut.width] !== labelId;
}

function writeLabelColor(
  pixels: Uint8ClampedArray,
  index: number,
  labelId: number,
  labels: AtlasLabel[],
) {
  if (labelId === 0) return;
  const rgb = labels[labelId]?.rgb;
  if (!rgb) return;
  const color = Number.parseInt(rgb.replace(/^#/, ""), 16);
  const offset = index * 4;
  pixels[offset] = color >> 16;
  pixels[offset + 1] = (color >> 8) & 255;
  pixels[offset + 2] = color & 255;
  pixels[offset + 3] = 255;
}

function warpCut(
  cut: AtlasCut,
  markers: Marker[],
  registrationWidth = cut.width,
  registrationHeight = cut.height,
): AtlasCut {
  const registeredWidth = Number.isFinite(registrationWidth) && registrationWidth > 0
    ? registrationWidth : cut.width;
  const registeredHeight = Number.isFinite(registrationHeight) && registrationHeight > 0
    ? registrationHeight : cut.height;
  const scaleX = cut.width / registeredWidth;
  const scaleY = cut.height / registeredHeight;
  const marginX = cut.width / 10;
  const marginY = cut.height / 10;
  const source: Point[] = [
    [-marginX, -marginY], [cut.width * 1.1, -marginY],
    [-marginX, cut.height * 1.1], [cut.width * 1.1, cut.height * 1.1],
  ];
  const destination: Point[] = source.map(([x, y]) => [x, y]);
  let triangles: Triangle[] = [[0, 1, 2], [1, 2, 3]];

  for (const marker of markers) {
    const sourcePoint: Point = [marker.x * scaleX, marker.y * scaleY];
    const destinationPoint: Point = [marker.nx * scaleX, marker.ny * scaleY];
    const remove = triangles
      .map((triangle, index) =>
        pointInCircumcircle(
          destination[triangle[0]], destination[triangle[1]],
          destination[triangle[2]], destinationPoint,
        ) ? index : -1,
      )
      .filter((index) => index >= 0);
    if (remove.length === 0) continue;

    const edgeCounts = new Map<string, { edge: [number, number]; count: number }>();
    for (const index of remove) {
      const [a, b, c] = triangles[index];
      for (const edge of [[a, b], [a, c], [b, c]] as Array<[number, number]>) {
        const ordered: [number, number] = edge[0] < edge[1] ? edge : [edge[1], edge[0]];
        const key = `${ordered[0]}:${ordered[1]}`;
        const existing = edgeCounts.get(key);
        edgeCounts.set(key, { edge: ordered, count: (existing?.count ?? 0) + 1 });
      }
    }
    const removed = new Set(remove);
    triangles = triangles.filter((_, index) => !removed.has(index));
    const vertex = destination.length;
    source.push(sourcePoint);
    destination.push(destinationPoint);
    for (const { edge, count } of edgeCounts.values()) {
      if (count === 1) triangles.push([edge[0], edge[1], vertex]);
    }
  }

  const ids = new Uint16Array(cut.ids.length);

  for (const [ia, ib, ic] of triangles) {
    const a = destination[ia], b = destination[ib], c = destination[ic];
    const sa = source[ia], sb = source[ib], sc = source[ic];
    const minX = Math.max(0, Math.floor(Math.min(a[0], b[0], c[0])));
    const maxX = Math.min(cut.width - 1, Math.ceil(Math.max(a[0], b[0], c[0])));
    const minY = Math.max(0, Math.floor(Math.min(a[1], b[1], c[1])));
    const maxY = Math.min(cut.height - 1, Math.ceil(Math.max(a[1], b[1], c[1])));
    const denominator = (b[1] - c[1]) * (a[0] - c[0]) +
      (c[0] - b[0]) * (a[1] - c[1]);
    if (denominator === 0) continue;

    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const u = ((b[1] - c[1]) * (x - c[0]) +
          (c[0] - b[0]) * (y - c[1])) / denominator;
        const v = ((c[1] - a[1]) * (x - c[0]) +
          (a[0] - c[0]) * (y - c[1])) / denominator;
        const w = 1 - u - v;
        if (u < 0 || v < 0 || w < 0) continue;
        const sourceX = Math.round(u * sa[0] + v * sb[0] + w * sc[0]);
        const sourceY = Math.round(u * sa[1] + v * sb[1] + w * sc[1]);
        if (
          sourceX < 0 || sourceX >= cut.width ||
          sourceY < 0 || sourceY >= cut.height
        ) continue;
        const labelId = cut.ids[sourceX + sourceY * cut.width];
        ids[x + y * cut.width] = labelId;
      }
    }
  }
  return { ids, width: cut.width, height: cut.height };
}

function pointInCircumcircle(a: Point, b: Point, c: Point, point: Point) {
  const a2 = squaredDistance(b, c);
  const b2 = squaredDistance(a, c);
  const c2 = squaredDistance(a, b);
  const fa = a2 * (b2 + c2 - a2);
  const fb = b2 * (c2 + a2 - b2);
  const fc = c2 * (a2 + b2 - c2);
  const denominator = fa + fb + fc;
  if (denominator === 0) return false;
  const centerScaled: Point = [
    fa * a[0] + fb * b[0] + fc * c[0],
    fa * a[1] + fb * b[1] + fc * c[1],
  ];
  const radiusScaled = squaredDistance(
    [a[0] * denominator, a[1] * denominator], centerScaled,
  );
  return squaredDistance(
    [point[0] * denominator, point[1] * denominator], centerScaled,
  ) < radiusScaled;
}

function squaredDistance(a: Point, b: Point) {
  return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
}
