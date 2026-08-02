// Bundle the user-selected augmented variants into one downloadable dataset
// (.zip): per-variant PNG frames + a manifest and an lmfao-ready pipeline
// config so the selection is reproducible with the real library.

import JSZip from "jszip";
import { configFor } from "./augmentations";
import type { TileData } from "./pipeline";

export interface ClipMeta {
  source: string;
  width: number;
  height: number;
  frameCount: number;
  fps: number;
}

function canvasToPngBlob(c: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    c.toBlob((b) => (b ? resolve(b) : reject(new Error("PNG encode failed"))), "image/png");
  });
}

const README = `# LMFAO augmented dataset

Created in-browser from a single source clip with the LMFAO augmentation demo.

Layout
  manifest.json      dataset descriptor + one entry per selected variant
  pipeline.json      lmfao pipeline config for the augmented variants
                     (feed to AugmentationPipeline.from_config)
  frames/<variant>/  per-variant PNG frames (frame_0000.png ...)

Each variant applies exactly ONE augmentation. The "augmentation" and
"config" fields in manifest.json name the exact lmfao augmenter and params
used, so every clip here is reproducible from the library.
`;

export async function exportDataset(
  selected: TileData[],
  meta: ClipMeta,
  onProgress?: (done: number, total: number) => void
): Promise<Blob> {
  const zip = new JSZip();
  const totalFrames = selected.reduce((n, t) => n + t.frames.length, 0);
  let done = 0;

  const variants: unknown[] = [];
  for (const t of selected) {
    const folder = zip.folder(`frames/${t.spec.id}`)!;
    for (let i = 0; i < t.frames.length; i++) {
      const blob = await canvasToPngBlob(t.frames[i]);
      folder.file(`frame_${String(i).padStart(4, "0")}.png`, blob);
      done++;
      onProgress?.(done, totalFrames);
    }
    variants.push({
      id: t.spec.id,
      label: t.spec.label,
      family: t.spec.family,
      augmentation: t.spec.registered,
      config: configFor(t.spec),
      frames: t.frames.length,
      dir: `frames/${t.spec.id}`,
    });
  }

  const manifest = {
    dataset: "lmfao-augmented",
    created_from: meta.source,
    resolution: { width: meta.width, height: meta.height },
    fps: Number(meta.fps.toFixed(2)),
    frame_count: meta.frameCount,
    variant_count: variants.length,
    variants,
  };
  zip.file("manifest.json", JSON.stringify(manifest, null, 2));

  const pipeline = selected.map((t) => configFor(t.spec)).filter(Boolean);
  zip.file("pipeline.json", JSON.stringify(pipeline, null, 2));
  zip.file("README.txt", README);

  return zip.generateAsync({ type: "blob", compression: "DEFLATE" });
}

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}
