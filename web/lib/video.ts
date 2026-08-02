// Decode a video (uploaded File or a URL) into a fixed set of downscaled RGBA
// frames, entirely in the browser. We seek to evenly-spaced timestamps and
// snapshot each into a canvas, then read back ImageData. Downscaling keeps
// memory sane when 16 tiles each hold their own augmented copy of the clip.

export interface ExtractedClip {
  frames: ImageData[];
  width: number;
  height: number;
  sourceWidth: number;
  sourceHeight: number;
  fps: number;
  duration: number;
}

export interface ExtractOptions {
  maxFrames?: number;
  maxWidth?: number;
}

const DEFAULTS = { maxFrames: 28, maxWidth: 320 };

function waitFor(el: HTMLVideoElement, event: string, timeoutMs = 12000): Promise<void> {
  return new Promise((resolve, reject) => {
    let done = false;
    const finish = (err?: Error) => {
      if (done) return;
      done = true;
      el.removeEventListener(event, ok);
      el.removeEventListener("error", bad);
      clearTimeout(timer);
      err ? reject(err) : resolve();
    };
    const ok = () => finish();
    const bad = () => finish(new Error(`video ${event} failed to load`));
    const timer = setTimeout(() => finish(new Error(`timed out waiting for ${event}`)), timeoutMs);
    el.addEventListener(event, ok, { once: true });
    el.addEventListener("error", bad, { once: true });
  });
}

function seek(el: HTMLVideoElement, time: number): Promise<void> {
  return new Promise((resolve, reject) => {
    let done = false;
    const finish = (err?: Error) => {
      if (done) return;
      done = true;
      el.removeEventListener("seeked", ok);
      clearTimeout(timer);
      err ? reject(err) : resolve();
    };
    const ok = () => finish();
    const timer = setTimeout(() => finish(), 4000); // resolve anyway; draw whatever is there
    el.addEventListener("seeked", ok, { once: true });
    try {
      el.currentTime = time;
    } catch (e) {
      finish(e as Error);
    }
  });
}

export async function extractFrames(
  source: File | string,
  opts: ExtractOptions = {}
): Promise<ExtractedClip> {
  const maxFrames = opts.maxFrames ?? DEFAULTS.maxFrames;
  const maxWidth = opts.maxWidth ?? DEFAULTS.maxWidth;

  const objectUrl = typeof source === "string" ? null : URL.createObjectURL(source);
  const src = objectUrl ?? (source as string);

  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  video.crossOrigin = "anonymous";
  video.src = src;

  try {
    await waitFor(video, "loadedmetadata");
    // Nudge decoding so the first frame is actually available to draw.
    await seek(video, 0);
    if (video.readyState < 2) {
      await waitFor(video, "loadeddata").catch(() => undefined);
    }

    const sourceWidth = video.videoWidth || 640;
    const sourceHeight = video.videoHeight || 360;
    const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 0;

    const scale = Math.min(1, maxWidth / sourceWidth);
    const width = Math.max(1, Math.round(sourceWidth * scale));
    const height = Math.max(1, Math.round(sourceHeight * scale));

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("2D canvas context unavailable");

    const frameCount = duration > 0 ? maxFrames : 1;
    const frames: ImageData[] = [];
    // Sample within [0, duration) leaving a small tail margin so the last seek
    // doesn't overshoot the end and return a blank frame.
    const span = duration > 0 ? duration * 0.98 : 0;
    for (let i = 0; i < frameCount; i++) {
      const t = frameCount > 1 ? (span * i) / (frameCount - 1) : 0;
      await seek(video, t);
      ctx.drawImage(video, 0, 0, width, height);
      frames.push(ctx.getImageData(0, 0, width, height));
    }

    const fps = duration > 0 ? frameCount / duration : 12;

    return { frames, width, height, sourceWidth, sourceHeight, fps, duration };
  } finally {
    video.src = "";
    video.removeAttribute("src");
    video.load();
    if (objectUrl) URL.revokeObjectURL(objectUrl);
  }
}
