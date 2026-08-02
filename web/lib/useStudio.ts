"use client";

import { useCallback, useRef, useState } from "react";
import { buildTiles, type TileData } from "./pipeline";
import { downloadBlob, exportJob } from "./exporter";
import { parseLeRobotFolder, type LeRobotDataset } from "./lerobot";
import { extractFrames } from "./video";

export type Status = "idle" | "decoding" | "augmenting" | "ready" | "error";

export const DEMO_SRC = "/demo.mp4";
export const FRAMES_DEFAULT = 28;
export const FRAMES_MIN = 2;
// No hard maximum: extraction and augmentation are cancellable, so the user
// can ask for as many frames as they want and abort if it drags. This is only
// the point past which we surface a "this may be heavy" hint in the UI.
export const FRAMES_HEAVY = 80;
// Overview playback rate (a slideshow across the whole clip).
export const OVERVIEW_FPS = 8;
// If the sampled rate (frames / duration) exceeds this, the frames are dense
// enough that real-time playback looks right, so default to it.
export const REALTIME_THRESHOLD_FPS = 15;
// Base seed written into exported job configs; the CLI derives per-episode
// seeds from it, so any fixed value keeps runs reproducible.
export const JOB_SEED = 42;

interface PreviewSource {
  source: File | string;
  name: string;
  window?: { from: number; to: number };
}

// All studio state + actions, shared by every skin of the UI (slop / non-slop).
export function useStudio() {
  const [status, setStatus] = useState<Status>("idle");
  const [tiles, setTiles] = useState<TileData[]>([]);
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [error, setError] = useState<string | null>(null);
  const [meta, setMeta] = useState<{ w: number; h: number; frames: number; fps: number } | null>(null);
  const [startTime, setStartTime] = useState(0);
  const [fileName, setFileName] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [exporting, setExporting] = useState<boolean>(false);
  const [frameCount, setFrameCount] = useState(FRAMES_DEFAULT);
  const [realtime, setRealtime] = useState(false);

  // LeRobot dataset mode: set when the user picks a dataset folder. The tile
  // grid then previews one episode/camera at a time, and export produces a CLI
  // job that augments the WHOLE dataset.
  const [dataset, setDataset] = useState<LeRobotDataset | null>(null);
  const [episodeIndex, setEpisodeIndex] = useState(0);
  const [camera, setCamera] = useState<string | null>(null);

  const frameCountRef = useRef(FRAMES_DEFAULT);
  const lastSourceRef = useRef<PreviewSource | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // The sampled rate: how many extracted frames each real second of the clip
  // maps to. This is what "real-time" playback runs at.
  const sampledFps = meta?.fps ?? OVERVIEW_FPS;
  const playbackFps = Math.max(1, realtime ? sampledFps : OVERVIEW_FPS);
  const frameDurationMs = 1000 / playbackFps;

  const run = useCallback(async (preview: PreviewSource) => {
    lastSourceRef.current = preview;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    const { signal } = controller;

    setError(null);
    setStatus("decoding");
    setTiles([]);
    setSelected(new Set());
    setFileName(preview.name);
    setProgress({ done: 0, total: frameCountRef.current });
    try {
      const clip = await extractFrames(preview.source, {
        maxFrames: frameCountRef.current,
        maxWidth: 320,
        signal,
        window: preview.window,
        onFrame: (done, total) => setProgress({ done, total }),
      });
      if (clip.frames.length === 0) throw new Error("No frames could be decoded from this file.");
      setMeta({ w: clip.width, h: clip.height, frames: clip.frames.length, fps: clip.fps });
      // Dense sampling -> real-time reads well, so default to it.
      setRealtime(clip.fps > REALTIME_THRESHOLD_FPS);
      setStatus("augmenting");
      setProgress({ done: 0, total: 0 });
      const built = await buildTiles(clip.frames, (done, total) => setProgress({ done, total }), signal);
      setTiles(built);
      setSelected(new Set(built.map((t) => t.spec.id)));
      setStartTime(performance.now());
      setStatus("ready");
    } catch (e) {
      if (signal.aborted || (e as { name?: string })?.name === "AbortError") {
        // User cancelled: quietly return to the idle state.
        setStatus("idle");
        setFileName(null);
        setTiles([]);
        setProgress({ done: 0, total: 0 });
      } else {
        setError(e instanceof Error ? e.message : String(e));
        setStatus("error");
      }
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
    }
  }, []);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const toggleRealtime = useCallback(() => {
    setRealtime((r) => !r);
    // Restart the shared clock so the new rate plays from frame 0 cleanly.
    setStartTime(performance.now());
  }, []);

  /** Preview one episode/camera of the loaded dataset. */
  const previewEpisode = useCallback(
    (ds: LeRobotDataset, epIdx: number, cam: string) => {
      const ep = ds.episodes[epIdx];
      const video = ep?.videos[cam];
      if (!ep || !video) {
        setError(`Episode ${epIdx} has no video for ${cam}.`);
        setStatus("error");
        return;
      }
      setEpisodeIndex(epIdx);
      setCamera(cam);
      run({
        source: video.file,
        name: `${ds.name} · ep ${ep.index} · ${cam.replace(/^observation\.images\./, "")}`,
        window: { from: video.from, to: video.to },
      });
    },
    [run]
  );

  /** Entry point for the dataset folder picker. */
  const onFolder = useCallback(
    async (files: FileList | null) => {
      if (!files || files.length === 0) return;
      setError(null);
      setStatus("decoding");
      setProgress({ done: 0, total: 0 });
      try {
        const ds = await parseLeRobotFolder(files);
        setDataset(ds);
        const cam = ds.cameras.find((c) => ds.episodes[0].videos[c]) ?? ds.cameras[0];
        previewEpisode(ds, 0, cam);
      } catch (e) {
        setDataset(null);
        setError(e instanceof Error ? e.message : String(e));
        setStatus("error");
      }
    },
    [previewEpisode]
  );

  const selectEpisode = useCallback(
    (epIdx: number) => {
      if (!dataset || !camera) return;
      previewEpisode(dataset, epIdx, camera);
    },
    [dataset, camera, previewEpisode]
  );

  const selectCamera = useCallback(
    (cam: string) => {
      if (!dataset) return;
      previewEpisode(dataset, episodeIndex, cam);
    },
    [dataset, episodeIndex, previewEpisode]
  );

  const onFile = useCallback(
    (file: File | undefined) => {
      if (!file) return;
      if (!file.type.startsWith("video/")) {
        setError("Please choose a video file.");
        setStatus("error");
        return;
      }
      setDataset(null);
      setCamera(null);
      run({ source: file, name: file.name });
    },
    [run]
  );

  const toggle = useCallback((id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const selectAll = useCallback(() => setSelected(new Set(tiles.map((t) => t.spec.id))), [tiles]);
  const clearAll = useCallback(() => setSelected(new Set()), []);

  const onFramesChange = useCallback((raw: number) => {
    // Clamp to a sane minimum only; no maximum (the run is cancellable).
    const clamped = Math.max(FRAMES_MIN, Math.round(raw || FRAMES_MIN));
    setFrameCount(clamped);
    frameCountRef.current = clamped;
  }, []);

  const regenerate = useCallback(() => {
    const s = lastSourceRef.current;
    if (s) run(s);
  }, [run]);

  const runDemo = useCallback(() => {
    setDataset(null);
    setCamera(null);
    run({ source: DEMO_SRC, name: "demo.mp4" });
  }, [run]);

  const onExport = useCallback(async () => {
    const chosen = tiles.filter((t) => selected.has(t.spec.id));
    if (chosen.length === 0) return;
    setExporting(true);
    try {
      const blob = await exportJob(chosen, {
        source: dataset?.name ?? fileName ?? "clip",
        cameras: dataset && camera ? [camera] : [],
        seed: JOB_SEED,
      });
      const stem = (dataset?.name ?? fileName ?? "clip").replace(/\.[^.]+$/, "").replace(/[^\w.-]+/g, "-");
      downloadBlob(blob, `${stem}-lmfao-job.zip`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setExporting(false);
    }
  }, [tiles, selected, dataset, camera, fileName]);

  const busy = status === "decoding" || status === "augmenting";

  return {
    status,
    tiles,
    progress,
    error,
    meta,
    startTime,
    fileName,
    selected,
    exporting,
    frameCount,
    frameDurationMs,
    realtime,
    sampledFps,
    playbackFps,
    busy,
    dataset,
    episodeIndex,
    camera,
    run,
    runDemo,
    cancel,
    toggleRealtime,
    onFile,
    onFolder,
    selectEpisode,
    selectCamera,
    toggle,
    selectAll,
    clearAll,
    onFramesChange,
    regenerate,
    onExport,
  };
}
