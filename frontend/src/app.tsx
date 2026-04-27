import { h } from "preact";
import { useEffect, useRef } from "preact/hooks";
import { useSignal, useComputed } from "@preact/signals";
import { DziViewer } from "./dzi_viewer";
import { BrushingCanvas, PredictionOverlay } from "./brushing_canvas";
import { ApiClient, SessionAllocatorClient, featureConfigToFilters } from "./api";
import * as state from "./state";
import { FeaturePanel } from "./components/FeaturePanel";
import { ControlBar } from "./components/ControlBar";
import { DataPanel } from "./components/DataPanel";
import { JobStatusPanel } from "./components/JobStatusPanel";
import type { TrainRequest } from "./types";
import { saveProjectToCloud } from "./state";

/** Parse URL query params once at startup into state signals. */
function readUrlParams() {
  const p = new URLSearchParams(window.location.search);
  const workdir = p.get("workdir");
  const token = p.get("token");
  const server = p.get("server");
  const allocator = p.get("allocator");
  if (workdir) state.workdir.value = workdir;
  if (token) state.bearerToken.value = token;
  if (server) state.serverUrl.value = server;
  if (allocator) state.allocatorUrl.value = allocator;
}

export function App() {
  const viewerContainerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<DziViewer | null>(null);
  const brushRef = useRef<BrushingCanvas | null>(null);
  const predRef = useRef<PredictionOverlay | null>(null);
  const saveStatus = useSignal<"idle" | "saving" | "saved" | "error">("idle");
  const saveMsg = useSignal<string>("");

  // Read URL params once
  useEffect(() => {
    readUrlParams();
  }, []);

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

    // Auto-load t_source DZIP if it was set via URL param
    const initialUrl = state.dziUrl.value;
    if (initialUrl) {
      handleLoadImage(initialUrl);
    }

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

  // Sync annotation panel hover highlight to brush canvas
  useEffect(() => {
    return state.highlightedStrokeIdx.subscribe((idx: number | null) => {
      brushRef.current?.setHighlightedStroke(idx);
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

      // Re-fire onViewChanged so the brushing canvas immediately redraws
      // strokes at the correct positions for the freshly loaded image's
      // pan/zoom, rather than waiting for the next user interaction.
      viewerRef.current!.onViewChanged?.();

      // If a classifier is already trained, re-point the prediction overlay at
      // the new image instead of discarding it. The tile URL function is
      // rebuilt with the new dzip_url / dzi_name while keeping the same
      // classifier_id, so predictions appear immediately on the new image.
      const existingId = state.classifierId.value;
      if (existingId) {
        const client = new ApiClient(state.serverUrl.value, state.bearerToken.value);
        const fc = state.featureConfig.value;
        const newDziName = viewerRef.current!.dziName ?? "";
        const tLevel = state.trainedLevel.value ?? state.workLevel.value ?? 0;
        predRef.current?.setLockedLevel(tLevel);
        predRef.current?.setTileUrlFn((level: number, col: number, row: number) =>
          client.predictionTileUrl({
            classifierId: existingId,
            level, col, row,
            dzipUrl: url,
            dziName: newDziName,
            featureConfig: fc,
          }),
        );
      } else {
        state.trainingStatus.value = "idle";
      }
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
    if (filters.length === 0) { alert("Select at least one feature filter."); return; }

    // Collect strokes from ALL annotated images (current + saved per-source)
    const allBySource: Record<string, import("./types").Stroke[]> = {
      ...state.strokesBySource.value,
    };
    if (state.dziUrl.value) {
      allBySource[state.dziUrl.value] = state.strokes.value;
    }
    const annotations = Object.entries(allBySource)
      .filter(([, ss]) => ss.length > 0)
      .map(([dzip_url, ss]) => ({
        dzip_url,
        level,
        strokes: ss.map((s) => ({
          label: s.labelId,
          points: s.points.map(([x, y]) => {
            const f = Math.pow(2, level - s.level);
            return [Math.round(x * f), Math.round(y * f)] as [number, number];
          }),
        })),
      }));

    if (annotations.length === 0) { alert("Add some brush strokes first."); return; }

    const req: import("./types").TrainMultiRequest = {
      annotations,
      features: { filters, scales: fc.scales },
    };

    state.trainingStatus.value = "training";
    state.trainingError.value = "";
    try {
      const client = new ApiClient(state.serverUrl.value, state.bearerToken.value);
      const res = await client.trainMulti(req);
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

  async function handleExport() {
    const outDir = state.outputDir.value;
    const srcDir = state.sourceDir.value;
    if (!outDir || !srcDir) {
      alert("Set ?workdir= in the URL to enable HPC export.");
      return;
    }
    const strokes = state.strokes.value;
    if (strokes.length === 0) { alert("No annotations — paint strokes first."); return; }

    const fc = state.featureConfig.value;
    state.exportStatus.value = "submitting";
    try {
      const client = new SessionAllocatorClient(
        state.allocatorUrl.value, state.bearerToken.value,
      );
      const res = await client.submitHeadlessJob({
        annotations: [{
          dzip_url: state.dziUrl.value,
          level: state.workLevel.value ?? state.dziMeta.value?.maxLevel,
          strokes: strokes.map((s) => {
            const wl = state.workLevel.value ?? state.dziMeta.value?.maxLevel ?? s.level;
            const f = Math.pow(2, wl - s.level);
            return {
              label: s.labelId,
              points: s.points.map(([x, y]) => [Math.round(x * f), Math.round(y * f)] as [number, number]),
            };
          }),
        }],
        features: { filters: featureConfigToFilters(fc), scales: fc.scales },
        p_source: srcDir,
        output_dir: outDir,
      });
      const record = {
        job_id: res.job_id, slurm_job_id: res.slurm_job_id,
        slurm_state: res.slurm_state, status: res.status,
        p_source: res.p_source, output_dir: res.output_dir, log_path: res.log_path,
        created_at: Date.now(), annotated_images: 1,
      };
      state.addHpcJob(record);
      state.activeHpcJob.value = record;
      state.exportStatus.value = `✅ SLURM ${res.slurm_job_id} submitted`;
    } catch (err) {
      state.exportStatus.value = `❌ ${err}`;
    }
  }

  async function handleSave() {
    saveStatus.value = "saving";
    saveMsg.value = "";
    try {
      const url = await saveProjectToCloud();
      saveStatus.value = "saved";
      saveMsg.value = `✓ Saved to ${url.split("/").pop()}`;
      setTimeout(() => { saveStatus.value = "idle"; saveMsg.value = ""; }, 4000);
    } catch (err) {
      saveStatus.value = "error";
      saveMsg.value = String(err);
    }
  }

  return (
    <div class="app-root">
      <aside class="sidebar">
        <div class="sidebar-scroll">
          <DataPanel onLoad={handleLoadImage} />
          <FeaturePanel />
          <ControlBar onTrain={handleTrain} onExport={handleExport} />
          <JobStatusPanel />
        </div>
        <div class="sidebar-footer">
          <button
            class="btn btn-save-cloud"
            onClick={handleSave}
            disabled={saveStatus.value === "saving"}
          >
            {saveStatus.value === "saving" ? "Saving…" : "☁️ Save project"}
          </button>
          {saveMsg.value && (
            <p class={saveStatus.value === "error" ? "error" : "status"}>
              {saveMsg.value}
            </p>
          )}
        </div>
      </aside>
      <div class="viewer-area" ref={viewerContainerRef} />
    </div>
  );
}
