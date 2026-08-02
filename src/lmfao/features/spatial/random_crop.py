from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lmfao.base import Augmenter, Metadata, Video
from lmfao.features.spatial.utils import metadata_params, pad_mode_for_numpy, sample_shift
from lmfao.registry import register_augmenter


@register_augmenter(
    "spatial.random_crop",
    tags=("spatial", "geometric", "crop"),
    description=(
        "Pads each edge and takes a random crop back to the original size, "
        "independently per frame, shifting each frame by a few pixels. "
        "Prevents the policy from memorizing absolute pixel positions and "
        "forces it to learn relative spatial relationships."
    ),
)
@dataclass
class RandomCrop(Augmenter):
    """Shift-via-pad-and-crop augmentation sampled independently per frame."""

    pad: int = 8
    pad_mode: str = "reflect"
    shift_x: int | None = None
    shift_y: int | None = None

    def __post_init__(self) -> None:
        self.pad = int(self.pad)
        if self.pad <= 0:
            raise ValueError("pad must be > 0")
        self.pad_mode = pad_mode_for_numpy(self.pad_mode)

    def apply(self, video: Video, metadata: Metadata, rng: np.random.Generator) -> tuple[Video, Metadata]:
        num_frames, height, width = video.shape[0], video.shape[1], video.shape[2]
        if self.pad >= height or self.pad >= width:
            raise ValueError(f"pad ({self.pad}) must be smaller than both height ({height}) and width ({width})")

        shift_x = sample_shift(self.shift_x, self.pad, num_frames, rng)
        shift_y = sample_shift(self.shift_y, self.pad, num_frames, rng)

        padded = np.pad(
            video,
            ((0, 0), (self.pad, self.pad), (self.pad, self.pad), (0, 0)),
            mode=self.pad_mode,
        )

        augmented = np.empty_like(video)
        for frame_index in range(num_frames):
            start_row = self.pad + int(shift_y[frame_index])
            start_col = self.pad + int(shift_x[frame_index])
            augmented[frame_index] = padded[
                frame_index,
                start_row : start_row + height,
                start_col : start_col + width,
                :,
            ]

        metadata.setdefault("augmentation_params", {})[self.name] = metadata_params(
            {
                "shift_x": shift_x.tolist(),
                "shift_y": shift_y.tolist(),
                "pad": self.pad,
                "pad_mode": self.pad_mode,
            }
        )
        return augmented, metadata
