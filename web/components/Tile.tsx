"use client";

import { useEffect, useRef } from "react";
import { FAMILY_COLORS, type Family } from "@/lib/augmentations";

export interface TileProps {
  label: string;
  registered: string;
  blurb: string;
  family: Family;
  frames: HTMLCanvasElement[];
  frameDurationMs: number;
  /** Shared clock origin (performance.now) so every tile plays in lock-step. */
  startTime: number;
  selected: boolean;
  onToggle: () => void;
}

export default function Tile({
  label,
  registered,
  blurb,
  family,
  frames,
  frameDurationMs,
  startTime,
  selected,
  onToggle,
}: TileProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const frameNumRef = useRef<HTMLSpanElement | null>(null);
  const color = FAMILY_COLORS[family];

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
        if (frameNumRef.current) frameNumRef.current.textContent = `frame ${idx}`;
        last = idx;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [frames, frameDurationMs, startTime]);

  return (
    <figure
      className={`tile${selected ? " is-selected" : ""}`}
      style={selected ? { outline: `2px solid ${color}`, borderColor: color } : undefined}
    >
      <button
        type="button"
        className="tile-media"
        onClick={onToggle}
        aria-pressed={selected}
        aria-label={`${selected ? "Deselect" : "Select"} ${label}`}
      >
        <canvas ref={canvasRef} />
        <span className="tile-frame" ref={frameNumRef}>
          frame 0
        </span>
        <span className="tile-badge" style={{ color, borderColor: `${color}66` }}>
          {family}
        </span>
        <span
          className={`tile-check${selected ? " on" : ""}`}
          style={selected ? { background: color, borderColor: color } : undefined}
        >
          {selected ? "✓" : ""}
        </span>
      </button>
      <figcaption className="tile-caption">
        <div className="tile-title">{label}</div>
        <code className="tile-registered">{registered}</code>
        <div className="tile-blurb">{blurb}</div>
      </figcaption>
    </figure>
  );
}
