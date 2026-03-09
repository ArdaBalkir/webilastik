import { h } from "preact";
import { useEffect, useRef } from "preact/hooks";
import { useSignal, useComputed } from "@preact/signals";
import { DziViewer } from "./dzi_viewer";
import { BrushingCanvas, PredictionOverlay } from "./brushing_canvas";
import { ApiClient, featureConfigToFilters } from "./api";
import * as state from "./state";
import { LabelPanel } from "./components/LabelPanel";
import { FeaturePanel } from "./components/FeaturePanel";
import { ControlBar } from "./components/ControlBar";
import { DataPanel } from "./components/DataPanel";
import type { TrainRequest } from "./types";

export function App() {
  const viewerContainerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<DziViewer | null>(null);
  const brushRef = useRef<BrushingCanvas | null>(null);
  const predRef = useRef<PredictionOverlay | null>(null);

  // Initialise canvas objects once the container div mounts
  useEffect(() => {
    const container = viewerContainerRef.current!;
    const viewer = new DziViewer(container);
    const pred = new PredictionOverlay(container, viewer);
    const brush = new BrushingCanvas(container, viewer);

    viewerRef.current = viewer;
    predRef.current = pred;
    brushRef.current = brush;

    brush.onStrokeFinished = (stroke) => {
      state.strokes.value = [...state.strokes.value, stroke];
    };

    return () => {
      viewer.destroy();
      pred.destroy();
    };
  }, []);

  // Keep viewer tile level capped to chosen working resolution
  useEffect(() => {
    return state.workLevel.subscribe((level: number | null) => {
      viewerRef.current?.setWorkLevel(level);
      // Resolution changed — trained classifier is now stale, clear overlay
      state.classifierId.value = null;
      state.trainedLevel.value = null;
      predRef.current?.setLockedLevel(null);
      predRef.current?.clearCache();
    });
  }, []);

  // Keep viewer tile level capped to chosen working resolution
  useEffect(() => {
    return state.workLevel.subscribe((level: number | null) => {
      viewerRef.current?.setWorkLevel(level);
      // Resolution changed — trained classifier is now stale, clear overlay
      state.classifierId.value = null;
      state.trainedLevel.value = null;
      predRef.current?.setLockedLevel(null);
      predRef.current?.clearCache();
    });
  }, []);

  // Keep brush mode in sync with state.toolMode
  useEffect(() => {
    const unsub = state.toolMode.subscribe((mode: string) => {
      brushRef.current?.setEnabled(mode === "brush");
      if (viewerRef.current) {
        viewerRef.current.canvas.style.pointerEvents =
          mode === "brush" ? "none" : "auto";
        viewerRef.current.canvas.style.cursor =
          mode === "brush" ? "default" : "grab";
      }
    });
    return unsub;
  }, []);

  // Keep brush size in sync
  useEffect(() => {
    return state.brushSize.subscribe((s: number) =>
      brushRef.current?.setBrushSize(s),
    );
  }, []);

  // Keep active label in sync
  useEffect(() => {
    return state.activeLabelId.subscribe((id: number) =>
      brushRef.current?.setActiveLabel(id),
    );
  }, []);

  // Keep labels on brush canvas in sync (for colours)
  useEffect(() => {
    return state.labels.subscribe((ls: typeof state.labels.value) => {
      brushRef.current?.setLabels(ls);
    });
  }, []);

  // Sync strokes from state into brushing canvas (handles project load)
  useEffect(() => {
    return state.strokes.subscribe((s: typeof state.strokes.value) => {
      brushRef.current?.setStrokes(s);
    });
  }, []);

  // Prediction overlay visibility/opacity
  useEffect(() => {
    return state.predictionVisible.subscribe((v: boolean) =>
      predRef.current?.setVisible(v),
    );
  }, []);
  useEffect(() => {
    return state.predictionOpacity.subscribe((o: number) =>
      predRef.current?.setOpacity(o),
    );
  }, []);

  // When classifierId changes, update the prediction tile URL function
  useEffect(() => {
    return state.classifierId.subscribe((id: string | null) => {
      if (!id) {
        predRef.current?.setTileUrlFn(() => "");
        return;
      }
      const client = new ApiClient(
        state.serverUrl.value,
        state.bearerToken.value,
      );
      const fc = state.featureConfig.value;
      const dzUrl = state.dziUrl.value;
      const dzName = state.dziName.value;
      const tLevel = state.trainedLevel.value ?? state.workLevel.value ?? 0;
      predRef.current?.setLockedLevel(tLevel);
      predRef.current?.setTileUrlFn((level: number, col: number, row: number) =>
        client.predictionTileUrl({
          classifierId: id,
          level,
          col,
          row,
          dzipUrl: dzUrl,
          dziName: dzName,
          featureConfig: fc,
        }),
      );
      predRef.current?.clearCache();
    });
  }, []);

  async function handleLoadImage(url: string) {
    state.isLoadingImage.value = true;
    state.loadError.value = "";
    try {
      const token = state.bearerToken.value;
      const extraHeaders: Record<string, string> = token
        ? { Authorization: `Bearer ${token}` }
        : {};
      const meta = await viewerRef.current!.loadDzip(
        url,
        undefined,
        Object.keys(extraHeaders).length ? extraHeaders : undefined,
      );
      state.dziMeta.value = meta;
      state.dziName.value = viewerRef.current!.dziName ?? "";
      state.dziUrl.value = url;
      state.isLoadingImage.value = false;
      state.classifierId.value = null;
      state.trainingStatus.value = "idle";
      predRef.current?.clearCache();
    } catch (err) {
      state.loadError.value = String(err);
      state.isLoadingImage.value = false;
    }
  }

  async function handleTrain() {
    const meta = state.dziMeta.value;
    if (!meta) return;
    const level = state.workLevel.value ?? meta.maxLevel;
    const fc = state.featureConfig.value;
    const filters = featureConfigToFilters(fc);
    if (filters.length === 0) {
      alert("Select at least one feature filter.");
      return;
    }
    const strokes = state.strokes.value;
    if (strokes.length === 0) {
      alert("Add some brush strokes first.");
      return;
    }

    const req: TrainRequest = {
      dzip_url: state.dziUrl.value,
      dzi_name: state.dziName.value,
      level,
      strokes: strokes.map((s: (typeof strokes)[0]) => ({
        label: s.labelId,
        // Rescale points from the stroke's capture level to the working level
        points: s.points.map(([x, y]) => {
          const factor = Math.pow(2, level - s.level);
          return [Math.round(x * factor), Math.round(y * factor)] as [
            number,
            number,
          ];
        }),
      })),
      features: { filters, scales: fc.scales },
    };

    state.trainingStatus.value = "training";
    state.trainingError.value = "";
    try {
      const client = new ApiClient(
        state.serverUrl.value,
        state.bearerToken.value,
      );
      const res = await client.train(req);
      state.classifierId.value = res.classifier_id;
      state.numClasses.value = res.num_classes;
      state.trainedLevel.value = level;
      state.trainingStatus.value = "ready";
      state.predictionVisible.value = true;
    } catch (err) {
      state.trainingStatus.value = "error";
      state.trainingError.value = String(err);
    }
  }

  return (
    <div class="app-root">
      <div class="viewer-area" ref={viewerContainerRef} />
      <aside class="sidebar">
        <h1 class="app-title">Webilastik 2.0</h1>
        <DataPanel onLoad={handleLoadImage} />
        <ControlBar onTrain={handleTrain} />
        <LabelPanel />
        <FeaturePanel />
      </aside>
    </div>
  );
}
