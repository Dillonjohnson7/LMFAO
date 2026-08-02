"use client";

import { type RefObject, useEffect, useRef } from "react";

// Drives one tile's looping canvas playback from a shared clock so every tile
// stays in lock-step. Shared by both the slop and non-slop tile components.
export function useFramePlayer(
  canvasRef: RefObject<HTMLCanvasElement | null>,
  frames: HTMLCanvasElement[],
  frameDurationMs: number,
  startTime: number,
  onIndex?: (idx: number) => void
) {
  const onIndexRef = useRef(onIndex);
  onIndexRef.current = onIndex;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || frames.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = frames[0].width;
    canvas.height = frames[0].height;

    let raf = 0;
    let last = -1;
    const total = frames.length;

    const tick = () => {
      const elapsed = performance.now() - startTime;
      const idx = Math.floor(elapsed / frameDurationMs) % total;
      if (idx !== last) {
        ctx.drawImage(frames[idx], 0, 0);
        onIndexRef.current?.(idx);
        last = idx;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [canvasRef, frames, frameDurationMs, startTime]);
}
