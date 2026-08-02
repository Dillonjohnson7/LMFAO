"use client";

import { type CSSProperties, useEffect, useRef } from "react";
import { FAMILY_COLORS } from "@/lib/augmentations";
import type { TileData } from "@/lib/pipeline";
import { useFramePlayer } from "@/lib/useFramePlayer";

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
  startTime,
  playbackLabel,
  isSelected,
  onToggle,
  onNavigate,
  onClose,
}: ViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const frameNumRef = useRef<HTMLSpanElement | null>(null);

  const tile = tiles[index];
  const total = tiles.length;
  const prev = () => onNavigate((index - 1 + total) % total);
  const next = () => onNavigate((index + 1) % total);

  useFramePlayer(canvasRef, tile?.frames ?? [], frameDurationMs, startTime, (idx) => {
    if (frameNumRef.current) frameNumRef.current.textContent = String(idx).padStart(2, "0");
  });

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowLeft") onNavigate((index - 1 + total) % total);
      else if (e.key === "ArrowRight") onNavigate((index + 1) % total);
    };
    window.addEventListener("keydown", onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [index, total, onNavigate, onClose]);

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
            <span className="viewer-frame">
              frame <span ref={frameNumRef}>00</span>
            </span>
          </div>
          <button className="viewer-nav next" aria-label="Next" onClick={next}>
            ›
          </button>
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
