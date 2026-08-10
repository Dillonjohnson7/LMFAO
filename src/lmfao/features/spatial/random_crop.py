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
        "Pads each edge and takes a random crop back to the original size. "
        "temporal_mode='per_frame' shifts each frame independently, which "
        "prevents memorizing absolute pixel positions but injects "
        "position-label noise at the shift scale. temporal_mode='constant' "
        "samples one shift for the whole episode, simulating a small camera "
        "remount between sessions without per-frame jitter."
    ),
)
@dataclass
class RandomCrop(Augmenter):
    """Shift-via-pad-and-crop augmentation."""

    pad: int = 8
    pad_mode: str = "reflect"
    shift_x: int | list[int] | None = None
    shift_y: int | list[int] | None = None
    temporal_mode: str = "per_frame"

    def __post_init__(self) -> None:
        self.pad = int(self.pad)
        if self.pad <= 0:
            raise ValueError("pad must be > 0")
        if self.temporal_mode not in ("per_frame", "constant"):
            raise ValueError(
                f"temporal_mode must be 'per_frame' or 'constant', got {self.temporal_mode!r}"
            )
        # Validate the user-facing name but keep it as given, so metadata records
        # the value the API accepts (e.g. "zero"), not numpy's internal
        # "constant"; the translation happens at the np.pad call site.
        pad_mode_for_numpy(self.pad_mode)

    def apply(self, video: Video, metadata: Metadata, rng: np.random.Generator) -> tuple[Video, Metadata]:
        num_frames, height, width = video.shape[0], video.shape[1], video.shape[2]
        if self.pad >= height or self.pad >= width:
            raise ValueError(f"pad ({self.pad}) must be smaller than both height ({height}) and width ({width})")

        shift_x = self.shift_x
        shift_y = self.shift_y
        if self.temporal_mode == "constant":
            # Sample once per episode; sample_shift broadcasts scalars to every
            # frame, so the whole clip gets one stable remount-style offset.
            if shift_x is None:
                shift_x = int(rng.integers(-self.pad, self.pad + 1))
            if shift_y is None:
                shift_y = int(rng.integers(-self.pad, self.pad + 1))

        shift_x_arr = sample_shift(shift_x, self.pad, num_frames, rng)
        shift_y_arr = sample_shift(shift_y, self.pad, num_frames, rng)

        padded = np.pad(
            video,
            ((0, 0), (self.pad, self.pad), (self.pad, self.pad), (0, 0)),
            mode=pad_mode_for_numpy(self.pad_mode),
        )

        augmented = np.empty_like(video)
        for frame_index in range(num_frames):
            start_row = self.pad + int(shift_y_arr[frame_index])
            start_col = self.pad + int(shift_x_arr[frame_index])
            augmented[frame_index] = padded[
                frame_index,
                start_row : start_row + height,
                start_col : start_col + width,
                :,
            ]

        metadata.setdefault("augmentation_params", {})[self.name] = metadata_params(
            {
                "shift_x": shift_x_arr.tolist(),
                "shift_y": shift_y_arr.tolist(),
                "pad": self.pad,
                "pad_mode": self.pad_mode,
                "temporal_mode": self.temporal_mode,
            }
        )
        return augmented, metadata
