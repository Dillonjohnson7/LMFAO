"use client";

import { type CSSProperties, useRef } from "react";
import { FAMILY_COLORS, type Family } from "@/lib/augmentations";
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
  onOpen: () => void;
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
  onOpen,
}: TileCleanProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const frameNumRef = useRef<HTMLSpanElement | null>(null);
  const color = FAMILY_COLORS[family];

  useFramePlayer(canvasRef, frames, frameDurationMs, startTime, (idx) => {
    if (frameNumRef.current) frameNumRef.current.textContent = String(idx).padStart(2, "0");
  });

  return (
    <figure className={`clean-tile${selected ? " sel" : ""}`} style={{ "--fam": color } as CSSProperties}>
      <div
        className="clean-thumb"
        role="button"
        tabIndex={0}
        aria-label={`Preview ${label}`}
        onClick={onOpen}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            onOpen();
          }
        }}
      >
        <canvas ref={canvasRef} />
        <span className="clean-frame">
          <span ref={frameNumRef}>00</span>
        </span>
        <button
          type="button"
          className="clean-box"
          aria-pressed={selected}
          aria-label={`${selected ? "Deselect" : "Select"} ${label}`}
          onClick={(e) => {
            e.stopPropagation();
            onToggle();
          }}
        />
        <span className="clean-expand" aria-hidden="true">
          Preview
        </span>
      </div>
      <figcaption className="clean-cap">
        <div className="clean-caphead">
          <span className="clean-num">{String(index).padStart(2, "0")}</span>
          <span className="clean-label">{label}</span>
          <span className="clean-fam">{family}</span>
        </div>
        <code className="clean-reg">{registered === "none" ? family : registered}</code>
        <p className="clean-blurb">{blurb}</p>
      </figcaption>
    </figure>
  );
}
