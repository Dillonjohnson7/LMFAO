from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from lmfao.base import Augmenter, Metadata, Video
from lmfao.registry import register_augmenter

_PAD_MODES = {
    "reflect": "reflect",
    "zero": "constant",
}


@register_augmenter(
    "spatial.random_crop",
    tags=("spatial", "geometric", "crop"),
    description=(
        "Pads each edge and takes a random crop back to the original size, "
        "independently per frame, shifting each frame by a few pixels. "
        "Prevents the policy from memorizing absolute pixel positions (e.g. "
        "'cube at pixel (312, 240)') and forces it to learn relative "
        "spatial relationships (e.g. 'cube near gripper') -- the standard "
        "regularizer behind DrQ/DrQv2/RAD-style visual RL and LeRobot's own "
        "image transforms."
    ),
)
@dataclass
class RandomCrop(Augmenter):
    """Shift-via-pad-and-crop augmentation.

    A fresh ``(shift_x, shift_y)`` pair is sampled independently for
    *every frame* in the clip -- not once per episode. That's what the
    evidence behind this augmentation (DrQ, DrQv2, RAD, LeRobot's own
    transforms) is actually built on: the goal isn't to simulate a
    specific physical phenomenon like camera-mount drift, it's to deny
    the network any consistent absolute-pixel-position signal to
    memorize. Per-episode sampling with pad=10 only gives ~400 distinct
    offsets total; per-frame sampling gives a different crop on every
    single training image, every single epoch, which is a much stronger
    regularizer.

    This doesn't corrupt temporal consistency for ACT-style policies:
    each frame is consumed independently by the ResNet backbone before
    any temporal attention happens, so by the time the transformer
    reasons over the sequence it's working with post-shift feature
    vectors, not raw pixels -- there's no "camera shake" for it to see.

    ``pad`` sets both the padding added to each edge before cropping and
    the maximum shift magnitude in either direction (an integer sampled
    from ``[-pad, pad]`` along each axis, per frame). The output is
    always exactly the input size -- no object is ever cropped out, and
    with the default ``pad_mode="reflect"`` no synthetic black border is
    introduced either, so every output pixel is real (or mirrored-real)
    image content.

    ``shift_x``/``shift_y``, if set, override sampling and apply that
    fixed offset to every frame -- useful for deterministic tests/demos,
    not for actual training use.
    """

    pad: int = 8
    pad_mode: str = "reflect"
    shift_x: Optional[int] = None
    shift_y: Optional[int] = None

    def apply(self, video: Video, metadata: Metadata, rng: np.random.Generator) -> tuple[Video, Metadata]:
        if self.pad <= 0:
            raise ValueError("pad must be > 0")
        if self.pad_mode not in _PAD_MODES:
            raise ValueError(f"pad_mode must be one of {sorted(_PAD_MODES)}, got {self.pad_mode!r}")

        num_frames, height, width = video.shape[0], video.shape[1], video.shape[2]
        if self.pad >= height or self.pad >= width:
            raise ValueError(
                f"pad ({self.pad}) must be smaller than both height ({height}) and width ({width})"
            )

        if self.shift_x is None:
            shift_x = rng.integers(-self.pad, self.pad + 1, size=num_frames)
        else:
            shift_x = np.full(num_frames, int(np.clip(self.shift_x, -self.pad, self.pad)))

        if self.shift_y is None:
            shift_y = rng.integers(-self.pad, self.pad + 1, size=num_frames)
        else:
            shift_y = np.full(num_frames, int(np.clip(self.shift_y, -self.pad, self.pad)))

        pad_width = ((0, 0), (self.pad, self.pad), (self.pad, self.pad), (0, 0))
        padded = np.pad(video, pad_width, mode=_PAD_MODES[self.pad_mode])

        augmented = np.empty_like(video)
        for frame_idx in range(num_frames):
            start_row = self.pad + int(shift_y[frame_idx])
            start_col = self.pad + int(shift_x[frame_idx])
            augmented[frame_idx] = padded[frame_idx, start_row:start_row + height, start_col:start_col + width, :]

        metadata.setdefault("augmentation_params", {})[self.name] = {
            "shift_x": shift_x.tolist(),
            "shift_y": shift_y.tolist(),
            "pad": self.pad,
            "pad_mode": self.pad_mode,
        }
        return augmented, metadata
