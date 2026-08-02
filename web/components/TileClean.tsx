"use client";

import { useRef } from "react";
import { type Family } from "@/lib/augmentations";
import { useFramePlayer } from "@/lib/useFramePlayer";

export interface TileCleanProps {
  index: number;
  label: string;
  registered: string;
  blurb: string;
  family: Family;
  frames: HTMLCanvasElement[];
  frameDurationMs: number;
  startTime: number;
  selected: boolean;
  onToggle: () => void;
}

export default function TileClean({
  index,
  label,
  registered,
  blurb,
  family,
  frames,
  frameDurationMs,
  startTime,
  selected,
  onToggle,
}: TileCleanProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const frameNumRef = useRef<HTMLSpanElement | null>(null);

  useFramePlayer(canvasRef, frames, frameDurationMs, startTime, (idx) => {
    if (frameNumRef.current) frameNumRef.current.textContent = String(idx).padStart(2, "0");
  });

  return (
    <figure className={`clean-tile${selected ? " sel" : ""}`}>
      <button
        type="button"
        className="clean-thumb"
        onClick={onToggle}
        aria-pressed={selected}
        aria-label={`${selected ? "Deselect" : "Select"} ${label}`}
      >
        <canvas ref={canvasRef} />
        <span className="clean-frame">
          <span ref={frameNumRef}>00</span>
        </span>
        <span className="clean-box" aria-hidden="true" />
      </button>
      <figcaption className="clean-cap">
        <div className="clean-caphead">
          <span className="clean-num">{String(index).padStart(2, "0")}</span>
          <span className="clean-label">{label}</span>
        </div>
        <code className="clean-reg">{registered === "none" ? family : registered}</code>
        <p className="clean-blurb">{blurb}</p>
      </figcaption>
    </figure>
  );
}
