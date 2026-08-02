// In-browser reimplementations of the lmfao augmentations, faithful to the
// numpy pixel math in src/lmfao/features/*. Each augmentation is a pure
// function (frames, seed) -> frames. Frames are RGBA ImageData at uint8; the
// Uint8ClampedArray backing store clamps to [0,255] and rounds for us, which
// matches the library's preserve_dtype clip+cast for uint8.
//
// Every tile in the grid runs exactly ONE augmentation — augmentations are
// never composed, matching the demo requirement.

import { choice, mulberry32, randint, uniform } from "./rng";

export type Family = "original" | "lighting" | "spatial" | "occlusion" | "noise";
export type AugFn = (frames: ImageData[], seed: number) => ImageData[];

export interface AugSpec {
  id: string;
  label: string;
  /** The real registered name in the lmfao library, e.g. "lighting.brightness". */
  registered: string;
  family: Family;
  blurb: string;
  /** Params that reproduce this variant with the real lmfao library. */
  params: Record<string, unknown>;
  fn: AugFn;
}

/** lmfao pipeline-config entry for a variant, or null for the untouched original. */
export function configFor(spec: AugSpec): { name: string; params: Record<string, unknown> } | null {
  if (spec.family === "original") return null;
  return { name: spec.registered, params: spec.params };
}

// ---------- helpers ----------

function clone(src: ImageData): ImageData {
  const out = new ImageData(src.width, src.height);
  out.data.set(src.data);
  return out;
}

function blank(width: number, height: number): ImageData {
  return new ImageData(width, height);
}

// numpy "reflect" mode: mirror without repeating the edge sample.
function reflectIndex(i: number, n: number): number {
  if (n === 1) return 0;
  const period = 2 * (n - 1);
  let m = ((i % period) + period) % period;
  return m < n ? m : period - m;
}

interface Box {
  top: number;
  left: number;
  height: number;
  width: number;
}

// Mirrors occlusion/utils.py sample_box.
function sampleBox(
  rng: () => number,
  H: number,
  W: number,
  areaRange: [number, number],
  aspectRange: [number, number]
): Box {
  const targetArea = uniform(rng, areaRange[0], areaRange[1]) * H * W;
  const aspect = uniform(rng, aspectRange[0], aspectRange[1]);
  const height = Math.min(Math.max(Math.round(Math.sqrt(targetArea / aspect)), 1), H);
  const width = Math.min(Math.max(Math.round(Math.sqrt(targetArea * aspect)), 1), W);
  const top = randint(rng, 0, H - height);
  const left = randint(rng, 0, W - width);
  return { top, left, height, width };
}

function fillRect(
  d: Uint8ClampedArray,
  W: number,
  box: { top: number; left: number; height: number; width: number },
  r: number,
  g: number,
  b: number
) {
  for (let y = box.top; y < box.top + box.height; y++) {
    let i = (y * W + box.left) * 4;
    for (let x = 0; x < box.width; x++) {
      d[i] = r;
      d[i + 1] = g;
      d[i + 2] = b;
      d[i + 3] = 255;
      i += 4;
    }
  }
}

// ---------- lighting ----------

function brightness(factor: number): AugFn {
  return (frames) =>
    frames.map((f) => {
      const out = clone(f);
      const d = out.data;
      for (let i = 0; i < d.length; i += 4) {
        d[i] = d[i] * factor;
        d[i + 1] = d[i + 1] * factor;
        d[i + 2] = d[i + 2] * factor;
      }
      return out;
    });
}

function contrast(factor: number): AugFn {
  const pivot = 127.5;
  return (frames) =>
    frames.map((f) => {
      const out = clone(f);
      const d = out.data;
      for (let i = 0; i < d.length; i += 4) {
        d[i] = (d[i] - pivot) * factor + pivot;
        d[i + 1] = (d[i + 1] - pivot) * factor + pivot;
        d[i + 2] = (d[i + 2] - pivot) * factor + pivot;
      }
      return out;
    });
}

function colorTemp(shift: number, intensity = 0.35): AugFn {
  const rGain = 1 + intensity * -shift;
  const bGain = 1 + intensity * shift;
  return (frames) =>
    frames.map((f) => {
      const out = clone(f);
      const d = out.data;
      for (let i = 0; i < d.length; i += 4) {
        d[i] = d[i] * rGain;
        d[i + 2] = d[i + 2] * bGain;
      }
      return out;
    });
}

// ---------- spatial ----------

function randomCrop(pad: number, mode: "reflect" | "zero"): AugFn {
  return (frames, seed) => {
    const rng = mulberry32(seed);
    return frames.map((f) => {
      const W = f.width;
      const H = f.height;
      const out = blank(W, H);
      const src = f.data;
      const dst = out.data;
      const sx = randint(rng, -pad, pad);
      const sy = randint(rng, -pad, pad);
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const di = (y * W + x) * 4;
          let srcY = y + sy;
          let srcX = x + sx;
          if (srcY < 0 || srcY >= H || srcX < 0 || srcX >= W) {
            if (mode === "reflect") {
              srcY = reflectIndex(srcY, H);
              srcX = reflectIndex(srcX, W);
            } else {
              dst[di] = 0;
              dst[di + 1] = 0;
              dst[di + 2] = 0;
              dst[di + 3] = 255;
              continue;
            }
          }
          const si = (srcY * W + srcX) * 4;
          dst[di] = src[si];
          dst[di + 1] = src[si + 1];
          dst[di + 2] = src[si + 2];
          dst[di + 3] = 255;
        }
      }
      return out;
    });
  };
}

// ---------- occlusion ----------

function sequenceBox(fill: "mean" | "black"): AugFn {
  return (frames, seed) => {
    const rng = mulberry32(seed);
    const W = frames[0].width;
    const H = frames[0].height;
    const box = sampleBox(rng, H, W, [0.02, 0.2], [0.5, 2.0]);

    let r = 0;
    let g = 0;
    let b = 0;
    if (fill === "mean") {
      let sr = 0;
      let sg = 0;
      let sb = 0;
      let n = 0;
      for (const f of frames) {
        const d = f.data;
        for (let y = box.top; y < box.top + box.height; y++) {
          let i = (y * W + box.left) * 4;
          for (let x = 0; x < box.width; x++) {
            sr += d[i];
            sg += d[i + 1];
            sb += d[i + 2];
            n += 1;
            i += 4;
          }
        }
      }
      r = Math.round(sr / n);
      g = Math.round(sg / n);
      b = Math.round(sb / n);
    }

    return frames.map((f) => {
      const out = clone(f);
      fillRect(out.data, W, box, r, g, b);
      return out;
    });
  };
}

type Edge = "top" | "bottom" | "left" | "right";

function borderIntrusion(
  edges: Edge[],
  maxFraction: number,
  mode: "constant" | "per_frame"
): AugFn {
  return (frames, seed) => {
    const rng = mulberry32(seed);
    const W = frames[0].width;
    const H = frames[0].height;

    const regionFor = (edge: Edge, fraction: number): Box => {
      if (edge === "top") {
        return { top: 0, left: 0, height: Math.max(1, Math.round(H * fraction)), width: W };
      }
      if (edge === "bottom") {
        const s = Math.max(1, Math.round(H * fraction));
        return { top: H - s, left: 0, height: s, width: W };
      }
      if (edge === "left") {
        return { top: 0, left: 0, height: H, width: Math.max(1, Math.round(W * fraction)) };
      }
      const s = Math.max(1, Math.round(W * fraction));
      return { top: 0, left: W - s, height: H, width: s };
    };

    let edge = choice(rng, edges);
    let fraction = uniform(rng, 0.02, maxFraction);

    return frames.map((f) => {
      if (mode === "per_frame") {
        edge = choice(rng, edges);
        fraction = uniform(rng, 0.02, maxFraction);
      }
      const out = clone(f);
      fillRect(out.data, W, regionFor(edge, fraction), 0, 0, 0);
      return out;
    });
  };
}

function positionAxis(
  start: number,
  v: number,
  t: number,
  maxPos: number,
  bounce: boolean
): number {
  if (maxPos <= 0) return 0;
  const raw = start + v * t;
  if (!bounce) return Math.min(Math.max(Math.round(raw), 0), maxPos);
  const period = 2 * maxPos;
  let wrapped = ((raw % period) + period) % period;
  if (wrapped > maxPos) wrapped = period - wrapped;
  return Math.round(wrapped);
}

function movingBox(fill: "random_color" | "black" | "mean"): AugFn {
  return (frames, seed) => {
    const rng = mulberry32(seed);
    const W = frames[0].width;
    const H = frames[0].height;
    const box = sampleBox(rng, H, W, [0.03, 0.18], [0.5, 2.0]);
    const vy = uniform(rng, -8, 8);
    const vx = uniform(rng, -8, 8);
    const maxY = H - box.height;
    const maxX = W - box.width;

    let mr = 0;
    let mg = 0;
    let mb = 0;
    if (fill === "mean") {
      let sr = 0;
      let sg = 0;
      let sb = 0;
      let n = 0;
      const stride = 4 * 251; // sparse sample of the whole clip for the mean color
      for (const f of frames) {
        const d = f.data;
        for (let i = 0; i < d.length; i += stride) {
          sr += d[i];
          sg += d[i + 1];
          sb += d[i + 2];
          n += 1;
        }
      }
      mr = Math.round(sr / n);
      mg = Math.round(sg / n);
      mb = Math.round(sb / n);
    }

    return frames.map((f, t) => {
      const out = clone(f);
      const top = positionAxis(box.top, vy, t, maxY, true);
      const left = positionAxis(box.left, vx, t, maxX, true);
      let r: number;
      let g: number;
      let b: number;
      if (fill === "random_color") {
        r = randint(rng, 0, 255);
        g = randint(rng, 0, 255);
        b = randint(rng, 0, 255);
      } else if (fill === "black") {
        r = 0;
        g = 0;
        b = 0;
      } else {
        r = mr;
        g = mg;
        b = mb;
      }
      fillRect(out.data, W, { top, left, height: box.height, width: box.width }, r, g, b);
      return out;
    });
  };
}

// ---------- noise ----------
// Mirrors noise/accelerator.py _add_noise_numpy for uint8: noise is resampled
// independently for every frame, pixel and channel; scaled by strength * 255
// (the uint8 dynamic range); added to the pixel; rounded then clipped.

// Box-Muller: one standard-normal sample from two uniforms.
function gauss(rng: () => number): number {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function gaussianNoise(sigma: number): AugFn {
  const scale = sigma * 255;
  return (frames, seed) => {
    const rng = mulberry32(seed);
    return frames.map((f) => {
      const out = clone(f);
      const d = out.data;
      for (let i = 0; i < d.length; i += 4) {
        d[i] = Math.round(d[i] + gauss(rng) * scale);
        d[i + 1] = Math.round(d[i + 1] + gauss(rng) * scale);
        d[i + 2] = Math.round(d[i + 2] + gauss(rng) * scale);
      }
      return out;
    });
  };
}

function uniformNoise(amplitude: number): AugFn {
  const scale = amplitude * 255;
  return (frames, seed) => {
    const rng = mulberry32(seed);
    return frames.map((f) => {
      const out = clone(f);
      const d = out.data;
      for (let i = 0; i < d.length; i += 4) {
        d[i] = Math.round(d[i] + (rng() - 0.5) * scale);
        d[i + 1] = Math.round(d[i + 1] + (rng() - 0.5) * scale);
        d[i + 2] = Math.round(d[i + 2] + (rng() - 0.5) * scale);
      }
      return out;
    });
  };
}

const identity: AugFn = (frames) => frames;

// ---------- the tile registry: 16 tiles, one augmentation each ----------

export const AUGMENTATIONS: AugSpec[] = [
  {
    id: "original",
    label: "Original",
    registered: "—",
    family: "original",
    blurb: "The uploaded clip, untouched.",
    params: {},
    fn: identity,
  },
  {
    id: "brightness-up",
    label: "Brightness ↑",
    registered: "lighting.brightness",
    family: "lighting",
    blurb: "Every channel × 1.35.",
    params: { factor: 1.35 },
    fn: brightness(1.35),
  },
  {
    id: "brightness-down",
    label: "Brightness ↓",
    registered: "lighting.brightness",
    family: "lighting",
    blurb: "Every channel × 0.6.",
    params: { factor: 0.6 },
    fn: brightness(0.6),
  },
  {
    id: "contrast-up",
    label: "Contrast ↑",
    registered: "lighting.contrast",
    family: "lighting",
    blurb: "Pushed away from mid-gray (× 1.6).",
    params: { factor: 1.6 },
    fn: contrast(1.6),
  },
  {
    id: "contrast-down",
    label: "Contrast ↓",
    registered: "lighting.contrast",
    family: "lighting",
    blurb: "Flattened toward mid-gray (× 0.6).",
    params: { factor: 0.6 },
    fn: contrast(0.6),
  },
  {
    id: "warm",
    label: "Warm",
    registered: "lighting.color_temperature",
    family: "lighting",
    blurb: "Red gain up, blue gain down (shift −1).",
    params: { shift: -1, intensity: 0.35 },
    fn: colorTemp(-1),
  },
  {
    id: "cool",
    label: "Cool",
    registered: "lighting.color_temperature",
    family: "lighting",
    blurb: "Blue gain up, red gain down (shift +1).",
    params: { shift: 1, intensity: 0.35 },
    fn: colorTemp(1),
  },
  {
    id: "gaussian-noise",
    label: "Gaussian noise",
    registered: "noise.gaussian",
    family: "noise",
    blurb: "Zero-mean grain, resampled every frame (σ 0.06).",
    params: { sigma: 0.06 },
    fn: gaussianNoise(0.06),
  },
  {
    id: "uniform-noise",
    label: "Uniform noise",
    registered: "noise.uniform",
    family: "noise",
    blurb: "Flat quantisation-style noise (amp 0.1).",
    params: { amplitude: 0.1 },
    fn: uniformNoise(0.1),
  },
  {
    id: "crop-reflect",
    label: "Random crop · reflect",
    registered: "spatial.random_crop",
    family: "spatial",
    blurb: "Per-frame ±8px jitter, reflected edges.",
    params: { pad: 8, pad_mode: "reflect" },
    fn: randomCrop(8, "reflect"),
  },
  {
    id: "crop-zero",
    label: "Random crop · zero-pad",
    registered: "spatial.random_crop",
    family: "spatial",
    blurb: "Per-frame ±12px jitter, black edges.",
    params: { pad: 12, pad_mode: "zero" },
    fn: randomCrop(12, "zero"),
  },
  {
    id: "sequence-box",
    label: "Sequence box",
    registered: "occlusion.sequence_box",
    family: "occlusion",
    blurb: "One static patch, filled with its mean color.",
    params: { box_area_range: [0.02, 0.2], fill: "mean" },
    fn: sequenceBox("mean"),
  },
  {
    id: "sequence-box-black",
    label: "Sequence box · black",
    registered: "occlusion.sequence_box",
    family: "occlusion",
    blurb: "One static black patch, fixed all clip.",
    params: { box_area_range: [0.02, 0.2], fill: "black" },
    fn: sequenceBox("black"),
  },
  {
    id: "border-intrusion",
    label: "Border intrusion",
    registered: "occlusion.border_intrusion",
    family: "occlusion",
    blurb: "A black bar creeps in from one edge.",
    params: { edges: ["bottom", "left", "right"], max_fraction: 0.25, fill: "black", temporal_mode: "constant" },
    fn: borderIntrusion(["bottom", "left", "right"], 0.25, "constant"),
  },
  {
    id: "border-intrusion-pf",
    label: "Border intrusion · per-frame",
    registered: "occlusion.border_intrusion",
    family: "occlusion",
    blurb: "Edge bar jumps edges every frame.",
    params: { edges: ["top", "bottom", "left", "right"], max_fraction: 0.25, fill: "black", temporal_mode: "per_frame" },
    fn: borderIntrusion(["top", "bottom", "left", "right"], 0.25, "per_frame"),
  },
  {
    id: "moving-box",
    label: "Moving box",
    registered: "occlusion.moving_box",
    family: "occlusion",
    blurb: "Box bounces around, recolored each frame.",
    params: { box_area_range: [0.03, 0.18], velocity_range: [-8, 8], fill: "random_color", edge_bounce: true },
    fn: movingBox("random_color"),
  },
  {
    id: "moving-box-black",
    label: "Moving box · black",
    registered: "occlusion.moving_box",
    family: "occlusion",
    blurb: "Black box bounces off the frame edges.",
    params: { box_area_range: [0.03, 0.18], velocity_range: [-8, 8], fill: "black", edge_bounce: true },
    fn: movingBox("black"),
  },
  {
    id: "moving-box-mean",
    label: "Moving box · mean fill",
    registered: "occlusion.moving_box",
    family: "occlusion",
    blurb: "Mean-color box bounces off the edges.",
    params: { box_area_range: [0.03, 0.18], velocity_range: [-8, 8], fill: "mean", edge_bounce: true },
    fn: movingBox("mean"),
  },
];

export const FAMILY_COLORS: Record<Family, string> = {
  original: "#8a8a93",
  lighting: "#fbbf24",
  spatial: "#38bdf8",
  occlusion: "#c084fc",
  noise: "#f472b6",
};

// A stable per-tile seed so occlusion positions / crop jitter don't reshuffle
// on every React render.
export function seedFor(index: number): number {
  return (index + 1) * 0x9e3779b1;
}
