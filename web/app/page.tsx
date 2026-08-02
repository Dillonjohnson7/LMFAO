"use client";

import Link from "next/link";
import { useRef } from "react";
import TileClean from "@/components/TileClean";
import { FRAMES_MAX, FRAMES_MIN, useStudio } from "@/lib/useStudio";

export default function Page() {
  const s = useStudio();
  const inputRef = useRef<HTMLInputElement | null>(null);

  const selectedCount = s.selected.size;
  const exportPct =
    s.exporting && s.exporting.total ? Math.round((s.exporting.done / s.exporting.total) * 100) : 0;

  return (
    <div className="clean">
      <header className="clean-masthead">
        <div className="clean-wordmark">
          lmfao<span>/augment</span>
        </div>
        <Link href="/slop" className="clean-altlink">
          slop version
        </Link>
      </header>

      <main className="clean-main">
        <section className="clean-intro">
          <p className="clean-kicker">Data augmentation, watched not guessed</p>
          <h1 className="clean-title">
            See every augmentation on your footage before you commit it to a training run.
          </h1>
          <p className="clean-standfirst">
            Load one demonstration clip. It is decoded in your browser and run through each
            augmenter in the library, one per panel, so you can judge the effect frame by frame.
            Keep the panels that look right and export them as a dataset.
          </p>

          <div className="clean-actions">
            <button className="clean-btn solid" onClick={() => inputRef.current?.click()} disabled={s.busy}>
              {s.busy ? "Working" : "Choose a video"}
            </button>
            <button className="clean-btn" onClick={s.runDemo} disabled={s.busy}>
              Use the sample clip
            </button>
            <label className="clean-frames">
              Frames per clip
              <input
                type="number"
                min={FRAMES_MIN}
                max={FRAMES_MAX}
                step={1}
                value={s.frameCount}
                disabled={s.busy}
                onChange={(e) => s.onFramesChange(Number(e.target.value))}
              />
            </label>
            {s.fileName && s.status === "ready" && (
              <button className="clean-btn quiet" onClick={s.regenerate} disabled={s.busy}>
                Regenerate at {s.frameCount}
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
            <p className="clean-status">
              {s.status === "decoding" && `Decoding ${s.fileName}`}
              {s.status === "augmenting" && `Rendering augmentations ${s.progress.done} of ${s.progress.total}`}
              {s.status === "ready" &&
                s.meta &&
                `${s.tiles.length} panels, ${s.meta.frames} frames each, ${s.meta.w}×${s.meta.h}px, from ${s.fileName}`}
            </p>
          )}
          {s.error && <p className="clean-status err">{s.error}</p>}
        </section>

        {s.status === "ready" && (
          <>
            <div className="clean-bar">
              <div className="clean-bar-count">
                {selectedCount} of {s.tiles.length} kept
              </div>
              <div className="clean-bar-actions">
                <button className="clean-textbtn" onClick={s.selectAll} disabled={!!s.exporting}>
                  Keep all
                </button>
                <span className="clean-sep" aria-hidden="true" />
                <button className="clean-textbtn" onClick={s.clearAll} disabled={!!s.exporting}>
                  Keep none
                </button>
                <button
                  className="clean-btn solid sm"
                  onClick={s.onExport}
                  disabled={selectedCount === 0 || !!s.exporting}
                >
                  {s.exporting ? `Exporting ${exportPct}%` : `Export dataset (${selectedCount})`}
                </button>
              </div>
            </div>

            <section className="clean-grid" aria-label="Augmented panels">
              {s.tiles.map((t, i) => (
                <TileClean
                  key={t.spec.id}
                  index={i}
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
          <section className="clean-empty">
            <div className="clean-empty-row">
              <div className="clean-empty-k">Lighting</div>
              <div className="clean-empty-v">Brightness, contrast, colour temperature.</div>
            </div>
            <div className="clean-empty-row">
              <div className="clean-empty-k">Noise</div>
              <div className="clean-empty-v">Gaussian grain and uniform, resampled every frame.</div>
            </div>
            <div className="clean-empty-row">
              <div className="clean-empty-k">Spatial</div>
              <div className="clean-empty-v">Per-frame random crop jitter.</div>
            </div>
            <div className="clean-empty-row">
              <div className="clean-empty-k">Occlusion</div>
              <div className="clean-empty-v">Sequence box, border intrusion, moving box.</div>
            </div>
          </section>
        )}
      </main>

      <footer className="clean-foot">
        <span>Runs entirely in the browser. The clip is never uploaded.</span>
        <span>Panels match the pixel behaviour of the lmfao library.</span>
      </footer>
    </div>
  );
}
