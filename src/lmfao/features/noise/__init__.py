"""Noise augmenters for robot demonstration images.

Each noise type is its own module. The sampled ones -- gaussian, uniform, shot
-- differ only in the distribution they draw from, and everything
device-specific for them lives in `accelerator.py`, so adding another means
naming a distribution, not writing GPU code. Blur and compression instead read
each pixel's neighbours, which no distribution can express; their machinery
lives in `utils.py` and runs on NumPy only.

Videos enter and leave as NumPy arrays regardless of where the work ran.
"""

from . import accelerator, blur, compression, gaussian, shot, uniform, utils
from .blur import DefocusBlur
from .compression import CompressionArtifacts
from .gaussian import GaussianNoise
from .shot import ShotNoise
from .uniform import UniformNoise

__all__ = [
    "accelerator",
    "blur",
    "compression",
    "gaussian",
    "shot",
    "uniform",
    "utils",
    "CompressionArtifacts",
    "DefocusBlur",
    "GaussianNoise",
    "ShotNoise",
    "UniformNoise",
]
