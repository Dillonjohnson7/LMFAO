"""Noise augmenters for robot demonstration images.

Each noise type is its own module and differs only in the distribution it draws
from. Everything device-specific lives in `accelerator.py`, so adding a new
noise type means naming a distribution, not writing GPU code.

Videos enter and leave as NumPy arrays regardless of where the sampling ran.
"""

from . import accelerator, gaussian, uniform
from .gaussian import GaussianNoise
from .uniform import UniformNoise

__all__ = [
    "accelerator",
    "gaussian",
    "uniform",
    "GaussianNoise",
    "UniformNoise",
]
