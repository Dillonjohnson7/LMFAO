from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimpleInpainter:
    """Fill reconstruction holes by iterative nearest-neighbour diffusion.

    Reference stand-in for ``cv2.inpaint`` / LaMa: each masked pixel is replaced
    by the average of its known 8-neighbours, repeated until the mask is filled
    or ``max_iterations`` is reached. Anything still unknown (fully enclosed by
    holes) falls back to the mean of the known pixels. Good enough for the small
    residual gaps novel-view rendering leaves; the real backend swaps in behind
    the :class:`~lmfao.miniworld.interfaces.Inpainter` interface.
    """

    max_iterations: int = 32

    def inpaint(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame)
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != frame.shape[:2]:
            raise ValueError("mask must match the frame's spatial shape")
        if not mask.any():
            return frame

        work = frame.astype(float)
        known = ~mask
        if not known.any():
            return frame  # nothing to fill from

        fill_value = work[known].mean(axis=0)
        work[mask] = fill_value  # seed so averaging is well-defined

        for _ in range(self.max_iterations):
            holes = ~known
            if not holes.any():
                break
            neighbour_sum = np.zeros_like(work)
            neighbour_cnt = np.zeros(work.shape[:2], dtype=float)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    shifted = np.roll(np.roll(work, dy, axis=0), dx, axis=1)
                    valid = np.roll(np.roll(known, dy, axis=0), dx, axis=1)
                    neighbour_sum += shifted * valid[:, :, None]
                    neighbour_cnt += valid
            fillable = holes & (neighbour_cnt > 0)
            if not fillable.any():
                break
            averaged = neighbour_sum[fillable] / neighbour_cnt[fillable][:, None]
            work[fillable] = averaged
            known = known | fillable

        result = np.clip(work + (0.5 if np.issubdtype(frame.dtype, np.integer) else 0.0), 0, None)
        return result.astype(frame.dtype)
