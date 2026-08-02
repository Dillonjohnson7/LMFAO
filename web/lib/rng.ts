// Small deterministic PRNG so every augmented tile is reproducible across
// renders (same seed -> same box positions, crop jitter, colors).
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function uniform(rng: () => number, lo: number, hi: number): number {
  return lo + (hi - lo) * rng();
}

// Inclusive integer in [lo, hi], matching numpy rng.integers(lo, hi+1).
export function randint(rng: () => number, lo: number, hi: number): number {
  return lo + Math.floor(rng() * (hi - lo + 1));
}

export function choice<T>(rng: () => number, items: readonly T[]): T {
  return items[Math.floor(rng() * items.length)];
}
