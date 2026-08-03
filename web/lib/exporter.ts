// Export the user's selection as an lmfao-augment job config.
//
// The browser is the preview surface, not the generator: producing a real
// LeRobot dataset (mp4 encoding + parquet + stats over every episode) is heavy
// and belongs to the Python CLI. What we download here is the contract between
// the two: a job.json whose variants[].pipeline is exactly what
// AugmentationPipeline.from_config takes, so the CLI reproduces the panels the
// user previewed, for every episode of the dataset, in true LeRobot format.

import JSZip from "jszip";
import { configFor } from "./augmentations";
import type { TileData } from "./pipeline";

export interface JobMeta {
  /** Dataset folder name (or clip name when previewing a bare video). */
  source: string;
  /** Camera keys to augment; empty when previewing a bare video. */
  cameras: string[];
  seed: number;
}

export interface LmfaoJob {
  lmfao_job: 1;
  seed: number;
  cameras: string[];
  variants: {
    id: string;
    suffix: string;
    label: string;
    pipeline: { name: string; params: Record<string, unknown>; probability: number }[];
  }[];
}

export function buildJob(selected: TileData[], meta: JobMeta): LmfaoJob {
  const variants = selected
    .map((t) => {
      const config = configFor(t.spec);
      if (!config) return null; // the untouched "original" tile: nothing to run
      return {
        id: t.spec.id,
        suffix: t.spec.id.replace(/-/g, "_"),
        label: t.spec.label,
        pipeline: [{ name: config.name, params: config.params, probability: 1.0 }],
      };
    })
    .filter((v): v is NonNullable<typeof v> => v !== null);

  return {
    lmfao_job: 1,
    seed: meta.seed,
    cameras: meta.cameras,
    variants,
  };
}

function readme(job: LmfaoJob, meta: JobMeta): string {
  const cams = job.cameras.length ? ` --cameras ${job.cameras.join(" ")}` : "";
  return [
    "LMFAO augmentation job",
    "======================",
    "",
    `Selected in the browser from: ${meta.source}`,
    `Variants: ${job.variants.map((v) => v.id).join(", ") || "(none)"}`,
    "",
    "Each variant becomes one augmented copy of EVERY episode in the source",
    "dataset, written as a new LeRobot v3.0 dataset (mp4 video + parquet).",
    "Robot state/actions are copied through unchanged.",
    "",
    "Run it with the lmfao CLI:",
    "",
    "  pip install \"lmfao[datasets]\"",
    `  lmfao-augment --input <dataset_dir> --output <output_dir> --config job.json${cams}`,
    "",
    "The variants' pipelines are exactly what lmfao's",
    "AugmentationPipeline.from_config takes, so the output matches the panels",
    "you previewed.",
    "",
  ].join("\n");
}

export async function exportJob(selected: TileData[], meta: JobMeta): Promise<Blob> {
  const job = buildJob(selected, meta);
  if (job.variants.length === 0) {
    throw new Error("Only the untouched original is selected; pick at least one augmentation.");
  }
  const zip = new JSZip();
  zip.file("job.json", JSON.stringify(job, null, 2));
  zip.file("README.txt", readme(job, meta));
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
