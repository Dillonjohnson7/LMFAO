"use client";

import Link from "next/link";
import { useRef } from "react";
import Tile from "@/components/Tile";
import { FRAMES_MIN, useStudio } from "@/lib/useStudio";

export default function SlopPage() {
  const s = useStudio();
  const inputRef = useRef<HTMLInputElement | null>(null);

  return (
    <main className="page">
      <header className="hero">
        <div className="brand">
          <span className="logo">LMFAO</span>
          <span className="tag">augmentation demo</span>
          <Link href="/" className="version-link">
            view refined version →
          </Link>
        </div>
        <h1>One clip in, a whole augmented fleet out.</h1>
        <p className="lede">
          Upload a single robot-demonstration clip. It fans out into a grid of variants,
          each running exactly <strong>one</strong> augmentation from the library (lighting,
          noise, spatial, and occlusion), all playing at once, computed live in your browser.
          Then curate the ones you like and download them as a new dataset.
        </p>

        <div className="controls">
          <button className="btn primary" onClick={() => inputRef.current?.click()} disabled={s.busy}>
            {s.busy ? "Working…" : "Upload a video"}
          </button>
          <button className="btn ghost" onClick={s.runDemo} disabled={s.busy}>
            Try the demo clip
          </button>

          <div className="field" title={`Frames extracted per clip (min ${FRAMES_MIN}, no maximum)`}>
            <label htmlFor="frames">Frames / clip</label>
            <input
              id="frames"
              type="number"
              min={FRAMES_MIN}
              step={1}
              value={s.frameCount}
              disabled={s.busy}
              onChange={(e) => s.onFramesChange(Number(e.target.value))}
            />
          </div>

          {s.busy && (
            <button className="btn ghost" onClick={s.cancel}>
              ✕ Cancel
            </button>
          )}

          {s.fileName && s.status === "ready" && (
            <button className="btn ghost" onClick={s.regenerate} disabled={s.busy}>
              ↻ Regenerate at {s.frameCount}
            </button>
          )}

          <input
            ref={inputRef}
            type="file"
            accept="video/*"
            hidden
            onChange={(e) => s.onFile(e.target.files?.[0])}
          />
        </div>

        {s.fileName && s.status !== "error" && (
          <div className="status-line">
            <span className="dot" data-status={s.status} />
            {s.status === "decoding" &&
              `Decoding ${s.fileName}… frame ${s.progress.done}/${s.progress.total}`}
            {s.status === "augmenting" && `Applying augmentations… ${s.progress.done}/${s.progress.total}`}
            {s.status === "ready" &&
              s.meta &&
              `${s.tiles.length} variants · ${s.meta.frames} frames · ${s.meta.w}×${s.meta.h}px · ${s.fileName}`}
          </div>
        )}

        {s.error && (
          <div className="status-line error">
            <span className="dot" data-status="error" /> {s.error}
          </div>
        )}
      </header>

      {s.status === "ready" && (
        <>
          <div className="toolbar">
            <div className="toolbar-count">
              <strong>{s.selected.size}</strong> of {s.tiles.length} selected
            </div>
            <div className="toolbar-actions">
              <button className="btn tiny ghost" onClick={s.toggleRealtime}>
                {s.realtime ? `⏱ Real-time ${Math.round(s.playbackFps)}fps` : "⏱ Overview 8fps"}
              </button>
              <button className="btn tiny ghost" onClick={s.selectAll} disabled={!!s.exporting}>
                Select all
              </button>
              <button className="btn tiny ghost" onClick={s.clearAll} disabled={!!s.exporting}>
                Clear
              </button>
              <button
                className="btn tiny primary"
                onClick={s.onExport}
                disabled={s.selected.size === 0 || !!s.exporting}
              >
                {s.exporting
                  ? `Exporting… ${s.exporting.total ? Math.round((s.exporting.done / s.exporting.total) * 100) : 0}%`
                  : `⬇ Download dataset (${s.selected.size})`}
              </button>
            </div>
          </div>

          <section className="grid" aria-label="Augmented variants">
            {s.tiles.map((t) => (
              <Tile
                key={t.spec.id}
                label={t.spec.label}
                registered={t.spec.registered}
                blurb={t.spec.blurb}
                family={t.spec.family}
                frames={t.frames}
                frameDurationMs={s.frameDurationMs}
                startTime={s.startTime}
                selected={s.selected.has(t.spec.id)}
                onToggle={() => s.toggle(t.spec.id)}
              />
            ))}
          </section>
        </>
      )}

      {s.status === "idle" && (
        <section className="empty">
          <div className="empty-card">
            <p>
              Nothing loaded yet. Hit <strong>Try the demo clip</strong> to see the fan-out,
              or upload your own <code>.mp4</code> / <code>.webm</code>.
            </p>
            <ul>
              <li><span className="swatch lighting" /> lighting: brightness, contrast, color temperature</li>
              <li><span className="swatch noise" /> noise: gaussian, uniform</li>
              <li><span className="swatch spatial" /> spatial: per-frame random crop jitter</li>
              <li><span className="swatch occlusion" /> occlusion: sequence box, border intrusion, moving box</li>
            </ul>
            <p className="empty-note">
              Tap variants to select/deselect (they all start selected), then download the ones
              you like as a single <code>.zip</code> dataset: PNG frames plus an lmfao-ready config.
            </p>
          </div>
        </section>
      )}

      <footer className="foot">
        Everything runs client-side, so your video never leaves the browser. Augmentations mirror
        the <code>lmfao</code> library&apos;s pixel math.
      </footer>
    </main>
  );
}
