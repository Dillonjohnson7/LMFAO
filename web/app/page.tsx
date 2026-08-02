"use client";

import { useCallback, useMemo, useRef, useState } from "react";
import Tile from "@/components/Tile";
import { buildTiles, type TileData } from "@/lib/pipeline";
import { downloadBlob, exportDataset } from "@/lib/exporter";
import { extractFrames } from "@/lib/video";

type Status = "idle" | "decoding" | "augmenting" | "ready" | "error";

const DEMO_SRC = "/demo.mp4";

export default function Page() {
  const [status, setStatus] = useState<Status>("idle");
  const [tiles, setTiles] = useState<TileData[]>([]);
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [error, setError] = useState<string | null>(null);
  const [meta, setMeta] = useState<{ w: number; h: number; frames: number; fps: number } | null>(null);
  const [startTime, setStartTime] = useState(0);
  const [fileName, setFileName] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [exporting, setExporting] = useState<{ done: number; total: number } | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const frameDurationMs = useMemo(() => 1000 / 8, []); // ~8 fps playback

  const run = useCallback(async (source: File | string, name: string) => {
    setError(null);
    setStatus("decoding");
    setTiles([]);
    setSelected(new Set());
    setFileName(name);
    try {
      const clip = await extractFrames(source, { maxFrames: 28, maxWidth: 320 });
      if (clip.frames.length === 0) throw new Error("No frames could be decoded from this file.");
      setMeta({ w: clip.width, h: clip.height, frames: clip.frames.length, fps: clip.fps });
      setStatus("augmenting");
      setProgress({ done: 0, total: 0 });
      const built = await buildTiles(clip.frames, (done, total) => setProgress({ done, total }));
      setTiles(built);
      // Start with everything selected — curate down to the ones you like.
      setSelected(new Set(built.map((t) => t.spec.id)));
      setStartTime(performance.now());
      setStatus("ready");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus("error");
    }
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
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }, []);

  const selectAll = useCallback(() => setSelected(new Set(tiles.map((t) => t.spec.id))), [tiles]);
  const clearAll = useCallback(() => setSelected(new Set()), []);

  const onExport = useCallback(async () => {
    if (!meta) return;
    const chosen = tiles.filter((t) => selected.has(t.spec.id));
    if (chosen.length === 0) return;
    setExporting({ done: 0, total: 0 });
    try {
      const blob = await exportDataset(
        chosen,
        {
          source: fileName ?? "clip",
          width: meta.w,
          height: meta.h,
          frameCount: meta.frames,
          fps: meta.fps,
        },
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

  return (
    <main className="page">
      <header className="hero">
        <div className="brand">
          <span className="logo">LMFAO</span>
          <span className="tag">augmentation demo</span>
        </div>
        <h1>One clip in, a whole augmented fleet out.</h1>
        <p className="lede">
          Upload a single robot-demonstration clip. It fans out into a grid of variants,
          each running exactly <strong>one</strong> augmentation from the library — lighting,
          noise, spatial, and occlusion — all playing at once, computed live in your browser.
          Then curate the ones you like and download them as a new dataset.
        </p>

        <div className="controls">
          <button className="btn primary" onClick={() => inputRef.current?.click()} disabled={busy}>
            {busy ? "Working…" : "Upload a video"}
          </button>
          <button className="btn ghost" onClick={() => run(DEMO_SRC, "demo.mp4")} disabled={busy}>
            Try the demo clip
          </button>
          <input
            ref={inputRef}
            type="file"
            accept="video/*"
            hidden
            onChange={(e) => onFile(e.target.files?.[0])}
          />
        </div>

        {fileName && status !== "error" && (
          <div className="status-line">
            <span className="dot" data-status={status} />
            {status === "decoding" && `Decoding frames from ${fileName}…`}
            {status === "augmenting" && `Applying augmentations… ${progress.done}/${progress.total}`}
            {status === "ready" &&
              meta &&
              `${tiles.length} variants · ${meta.frames} frames · ${meta.w}×${meta.h}px · ${fileName}`}
          </div>
        )}

        {error && (
          <div className="status-line error">
            <span className="dot" data-status="error" /> {error}
          </div>
        )}
      </header>

      {status === "ready" && (
        <>
          <div className="toolbar">
            <div className="toolbar-count">
              <strong>{selected.size}</strong> of {tiles.length} selected
            </div>
            <div className="toolbar-actions">
              <button className="btn tiny ghost" onClick={selectAll} disabled={!!exporting}>
                Select all
              </button>
              <button className="btn tiny ghost" onClick={clearAll} disabled={!!exporting}>
                Clear
              </button>
              <button
                className="btn tiny primary"
                onClick={onExport}
                disabled={selected.size === 0 || !!exporting}
              >
                {exporting
                  ? `Exporting… ${exporting.total ? Math.round((exporting.done / exporting.total) * 100) : 0}%`
                  : `⬇ Download dataset (${selected.size})`}
              </button>
            </div>
          </div>

          <section className="grid" aria-label="Augmented variants">
            {tiles.map((t) => (
              <Tile
                key={t.spec.id}
                label={t.spec.label}
                registered={t.spec.registered}
                blurb={t.spec.blurb}
                family={t.spec.family}
                frames={t.frames}
                frameDurationMs={frameDurationMs}
                startTime={startTime}
                selected={selected.has(t.spec.id)}
                onToggle={() => toggle(t.spec.id)}
              />
            ))}
          </section>
        </>
      )}

      {status === "idle" && (
        <section className="empty">
          <div className="empty-card">
            <p>
              Nothing loaded yet. Hit <strong>Try the demo clip</strong> to see the fan-out,
              or upload your own <code>.mp4</code> / <code>.webm</code>.
            </p>
            <ul>
              <li><span className="swatch lighting" /> lighting — brightness, contrast, color temperature</li>
              <li><span className="swatch noise" /> noise — gaussian, uniform</li>
              <li><span className="swatch spatial" /> spatial — per-frame random crop jitter</li>
              <li><span className="swatch occlusion" /> occlusion — sequence box, border intrusion, moving box</li>
            </ul>
            <p className="empty-note">
              Tap variants to select/deselect (they all start selected), then download the ones
              you like as a single <code>.zip</code> dataset — PNG frames plus an lmfao-ready config.
            </p>
          </div>
        </section>
      )}

      <footer className="foot">
        Everything runs client-side — your video never leaves the browser. Augmentations mirror
        the <code>lmfao</code> library&apos;s pixel math.
      </footer>
    </main>
  );
}
