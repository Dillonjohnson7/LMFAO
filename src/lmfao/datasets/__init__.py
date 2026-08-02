"""Episode I/O layer (the keystone described in the v2 plan, §8a).

Nothing else in ``lmfao`` reads or writes episodes; the augmentation core is
numpy-only and works on bare ``(F, H, W, C)`` arrays. Both halves of the program
need one shared representation of a recorded (or manufactured) demonstration:
the v1 pipeline needs it to offer an end-to-end "augment this dataset" entry
point, and miniworld needs it to emit synthetic episodes in the same shape as
real ones.

``Episode`` is that representation. It is deliberately in-memory and numpy-only
so it never drags heavy dependencies (torch, video codecs) into the core; a
LeRobot reader/writer that populates it can live behind an optional extra.
"""

from lmfao.datasets.episode import Episode, stack_frames

__all__ = ["Episode", "stack_frames"]
