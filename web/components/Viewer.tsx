"use client";

import { type CSSProperties, useEffect, useRef, useState } from "react";
import { FAMILY_COLORS } from "@/lib/augmentations";
import type { TileData } from "@/lib/pipeline";

export interface ViewerProps {
  tiles: TileData[];
  index: number;
  frameDurationMs: number;
  startTime: number;
  playbackLabel: string;
  isSelected: boolean;
  onToggle: (id: string) => void;
  onNavigate: (nextIndex: number) => void;
  onClose: () => void;
}

export default function Viewer({
  tiles,
  index,
  frameDurationMs,
  playbackLabel,
  isSelected,
  onToggle,
  onNavigate,
  onClose,
}: ViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const clockRef = useRef(0); // local playback clock origin
  const [playing, setPlaying] = useState(true);
  const [frame, setFrame] = useState(0);

  const tile = tiles[index];
  const total = tiles.length;
  const nFrames = tile?.frames.length ?? 0;

  const prev = () => onNavigate((index - 1 + total) % total);
  const next = () => onNavigate((index + 1) % total);

  // Auto-play loop (runs only while `playing`).
  useEffect(() => {
    const canvas = canvasRef.current;
    const frames = tile?.frames ?? [];
    if (!canvas || frames.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = frames[0].width;
    canvas.height = frames[0].height;
    if (!playing) return;

    clockRef.current = performance.now() - frame * frameDurationMs;
    let raf = 0;
    let last = -1;
    const tick = () => {
      const idx = Math.floor((performance.now() - clockRef.current) / frameDurationMs) % frames.length;
      if (idx !== last) {
        ctx.drawImage(frames[idx], 0, 0);
        setFrame(idx);
        last = idx;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // `frame` is intentionally excluded: it changes every tick and would restart the loop.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tile, frameDurationMs, playing]);

  // Paused / scrubbed draw: show exactly the selected frame.
  useEffect(() => {
    if (playing) return;
    const canvas = canvasRef.current;
    const frames = tile?.frames ?? [];
    if (!canvas || frames.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = frames[0].width;
    canvas.height = frames[0].height;
    ctx.drawImage(frames[Math.min(frame, frames.length - 1)], 0, 0);
  }, [frame, playing, tile]);

  // Keyboard: Esc close, arrows navigate, space play/pause.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowLeft") prev();
      else if (e.key === "ArrowRight") next();
      else if (e.key === " ") {
        e.preventDefault();
        setPlaying((p) => !p);
      }
    };
    window.addEventListener("keydown", onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [index, total]);

  if (!tile) return null;
  const color = FAMILY_COLORS[tile.spec.family];

  return (
    <div className="viewer" role="dialog" aria-modal="true" aria-label={`Preview: ${tile.spec.label}`}>
      <button className="viewer-scrim" aria-label="Close preview" onClick={onClose} />
      <div className="viewer-panel" style={{ "--fam": color } as CSSProperties}>
        <div className="viewer-stage">
          <button className="viewer-nav prev" aria-label="Previous" onClick={prev}>
            ‹
          </button>
          <div className="viewer-canvas-wrap">
            <canvas ref={canvasRef} />
          </div>
          <button className="viewer-nav next" aria-label="Next" onClick={next}>
            ›
          </button>

          <div className="viewer-scrub">
            <button
              className="viewer-play"
              aria-label={playing ? "Pause" : "Play"}
              onClick={() => setPlaying((p) => !p)}
            >
              {playing ? "❚❚" : "▶"}
            </button>
            <input
              className="viewer-range"
              type="range"
              min={0}
              max={Math.max(0, nFrames - 1)}
              value={frame}
              onChange={(e) => {
                setPlaying(false);
                setFrame(Number(e.target.value));
              }}
              aria-label="Scrub frames"
            />
            <span className="viewer-scrubcount">
              {String(frame).padStart(2, "0")} / {nFrames - 1}
            </span>
          </div>
        </div>

        <aside className="viewer-meta">
          <div className="viewer-top">
            <span className="viewer-fam">{tile.spec.family}</span>
            <button className="viewer-close" aria-label="Close" onClick={onClose}>
              ✕
            </button>
          </div>
          <h2 className="viewer-title">{tile.spec.label}</h2>
          <code className="viewer-reg">{tile.spec.registered}</code>
          <p className="viewer-blurb">{tile.spec.blurb}</p>

          <dl className="viewer-params">
            <dt>Playback</dt>
            <dd>{playbackLabel}</dd>
            <dt>Panel</dt>
            <dd>
              {index + 1} of {total}
            </dd>
            {Object.keys(tile.spec.params).length > 0 && (
              <>
                <dt>Params</dt>
                <dd>
                  <code>{JSON.stringify(tile.spec.params)}</code>
                </dd>
              </>
            )}
          </dl>

          <button
            className={`viewer-keep${isSelected ? " on" : ""}`}
            onClick={() => onToggle(tile.spec.id)}
          >
            {isSelected ? "Selected · click to remove" : "Not selected · click to keep"}
          </button>
        </aside>
      </div>
    </div>
  );
}
