// Turn one set of base frames into per-tile augmented canvas frames. Each tile
// runs exactly one augmentation. Augmented ImageData is baked into small
// canvases up front so playback is just cheap drawImage calls.

import { AUGMENTATIONS, type AugSpec, seedFor } from "./augmentations";

export interface TileData {
  spec: AugSpec;
  frames: HTMLCanvasElement[];
}

function toCanvas(img: ImageData): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = img.width;
  c.height = img.height;
  const ctx = c.getContext("2d");
  if (!ctx) throw new Error("2D canvas context unavailable");
  ctx.putImageData(img, 0, 0);
  return c;
}

const nextFrame = () =>
  new Promise<void>((r) => requestAnimationFrame(() => r()));

export async function buildTiles(
  base: ImageData[],
  onProgress?: (done: number, total: number) => void
): Promise<TileData[]> {
  const tiles: TileData[] = [];
  const total = AUGMENTATIONS.length;
  for (let i = 0; i < total; i++) {
    const spec = AUGMENTATIONS[i];
    const augmented = spec.fn(base, seedFor(i));
    tiles.push({ spec, frames: augmented.map(toCanvas) });
    onProgress?.(i + 1, total);
    // Yield to the event loop so the progress UI can paint between tiles.
    await nextFrame();
  }
  return tiles;
}
