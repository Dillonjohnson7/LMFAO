"use client";

import Link from "next/link";
import { useRef, useState } from "react";
import TileClean from "@/components/TileClean";
import Viewer from "@/components/Viewer";
import { FRAMES_HEAVY, FRAMES_MIN, useStudio } from "@/lib/useStudio";

export default function Page() {
  const s = useStudio();
  const inputRef = useRef<HTMLInputElement | null>(null);
  const folderRef = useRef<HTMLInputElement | null>(null);
  const [viewerIndex, setViewerIndex] = useState<number | null>(null);

  const selectedCount = s.selected.size;
  // Only count real augmentations toward the job (the untouched original
  // panel exports nothing).
  const jobCount = s.tiles.filter((t) => s.selected.has(t.spec.id) && t.spec.family !== "original").length;

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
            Load a LeRobot dataset folder (or a single clip). Real episode frames are decoded in
            your browser and run through each augmenter in the library, one per panel, so you can
            judge the effect frame by frame. Keep the panels that look right and export a job
            config; the <code>lmfao-augment</code> CLI then applies it to every episode and writes
            a true LeRobot dataset.
          </p>

          <div className="clean-actions">
            <button className="clean-btn solid" onClick={() => folderRef.current?.click()} disabled={s.busy}>
              {s.busy ? "Working" : "Choose a dataset folder"}
            </button>
            <button className="clean-btn" onClick={() => inputRef.current?.click()} disabled={s.busy}>
              Single video
            </button>
            <button className="clean-btn" onClick={s.runDemo} disabled={s.busy}>
              Use the sample clip
            </button>
            <label className="clean-frames">
              Frames per clip
              <input
                type="number"
                min={FRAMES_MIN}
                step={1}
                value={s.frameCount}
                disabled={s.busy}
                onChange={(e) => s.onFramesChange(Number(e.target.value))}
              />
            </label>
            {s.busy && (
              <button className="clean-btn quiet" onClick={s.cancel}>
                Cancel
              </button>
            )}
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
            <input
              ref={folderRef}
              type="file"
              hidden
              // Non-standard but universally supported attribute for picking a
              // whole directory; not in React's TS types.
              {...({ webkitdirectory: "", directory: "" } as Record<string, string>)}
              onChange={(e) => s.onFolder(e.target.files)}
            />
          </div>

          {s.dataset && (
            <div className="clean-actions" style={{ marginTop: "0.75rem" }}>
              <label className="clean-frames">
                Episode
                <select
                  value={s.episodeIndex}
                  disabled={s.busy}
                  onChange={(e) => s.selectEpisode(Number(e.target.value))}
                >
                  {s.dataset.episodes.map((ep, i) => (
                    <option key={ep.index} value={i}>
                      {ep.index} ({ep.length} frames)
                    </option>
                  ))}
                </select>
              </label>
              {s.dataset.cameras.length > 1 && (
                <label className="clean-frames">
                  Camera
                  <select
                    value={s.camera ?? ""}
                    disabled={s.busy}
                    onChange={(e) => s.selectCamera(e.target.value)}
                  >
                    {s.dataset.cameras.map((c) => (
                      <option key={c} value={c}>
                        {c.replace(/^observation\.images\./, "")}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              <span className="clean-hint" style={{ margin: 0 }}>
                {s.dataset.episodes.length} episodes · the exported job augments all of them
              </span>
            </div>
          )}

          {!s.busy && s.frameCount > FRAMES_HEAVY && (
            <p className="clean-hint">
              {s.frameCount} frames is a lot: decoding is sequential and each panel keeps its own
              copy, so this can be slow and memory-heavy. You can cancel mid-run.
            </p>
          )}

          {s.fileName && s.status !== "error" && (
            <p className="clean-status">
              {s.status === "decoding" &&
                `Decoding ${s.fileName}, frame ${s.progress.done} of ${s.progress.total}`}
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
                <button
                  className="clean-textbtn playback"
                  onClick={s.toggleRealtime}
                  title="Switch between real-time and overview playback"
                >
                  {s.realtime
                    ? `real-time, ${Math.round(s.playbackFps)} fps`
                    : `overview, ${Math.round(s.playbackFps)} fps`}
                </button>
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
                  disabled={jobCount === 0 || s.exporting}
                >
                  {s.exporting ? "Exporting…" : `Export CLI job (${jobCount})`}
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
                  onOpen={() => setViewerIndex(i)}
                />
              ))}
            </section>
          </>
        )}

        {viewerIndex !== null && s.tiles[viewerIndex] && (
          <Viewer
            tiles={s.tiles}
            index={viewerIndex}
            frameDurationMs={s.frameDurationMs}
            startTime={s.startTime}
            playbackLabel={
              s.realtime
                ? `real-time, ${Math.round(s.playbackFps)} fps`
                : `overview, ${Math.round(s.playbackFps)} fps`
            }
            isSelected={s.selected.has(s.tiles[viewerIndex].spec.id)}
            onToggle={s.toggle}
            onNavigate={setViewerIndex}
            onClose={() => setViewerIndex(null)}
          />
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
