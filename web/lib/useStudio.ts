"use client";

import { useCallback, useRef, useState } from "react";
import { buildTiles, type TileData } from "./pipeline";
import { downloadBlob, exportDataset } from "./exporter";
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
  const [exporting, setExporting] = useState<{ done: number; total: number } | null>(null);
  const [frameCount, setFrameCount] = useState(FRAMES_DEFAULT);
  const [realtime, setRealtime] = useState(false);

  const frameCountRef = useRef(FRAMES_DEFAULT);
  const lastSourceRef = useRef<{ source: File | string; name: string } | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // The sampled rate: how many extracted frames each real second of the clip
  // maps to. This is what "real-time" playback runs at.
  const sampledFps = meta?.fps ?? OVERVIEW_FPS;
  const playbackFps = Math.max(1, realtime ? sampledFps : OVERVIEW_FPS);
  const frameDurationMs = 1000 / playbackFps;

  const run = useCallback(async (source: File | string, name: string) => {
    lastSourceRef.current = { source, name };
    const controller = new AbortController();
    abortRef.current = controller;
    const { signal } = controller;

    setError(null);
    setStatus("decoding");
    setTiles([]);
    setSelected(new Set());
    setFileName(name);
    setProgress({ done: 0, total: frameCountRef.current });
    try {
      const clip = await extractFrames(source, {
        maxFrames: frameCountRef.current,
        maxWidth: 320,
        signal,
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

  const onFile = useCallback(
    (file: File | undefined) => {
      if (!file) return;
      if (!file.type.startsWith("video/")) {
        setError("Please choose a video file.");
        setStatus("error");
        return;
      }
      run(file, file.name);
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
    if (s) run(s.source, s.name);
  }, [run]);

  const runDemo = useCallback(() => run(DEMO_SRC, "demo.mp4"), [run]);

  const onExport = useCallback(async () => {
    if (!meta) return;
    const chosen = tiles.filter((t) => selected.has(t.spec.id));
    if (chosen.length === 0) return;
    setExporting({ done: 0, total: 0 });
    try {
      const blob = await exportDataset(
        chosen,
        { source: fileName ?? "clip", width: meta.w, height: meta.h, frameCount: meta.frames, fps: meta.fps },
        (done, total) => setExporting({ done, total })
      );
      const stem = (fileName ?? "clip").replace(/\.[^.]+$/, "");
      downloadBlob(blob, `${stem}-lmfao-dataset.zip`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setExporting(null);
    }
  }, [tiles, selected, meta, fileName]);

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
    run,
    runDemo,
    cancel,
    toggleRealtime,
    onFile,
    toggle,
    selectAll,
    clearAll,
    onFramesChange,
    regenerate,
    onExport,
  };
}
