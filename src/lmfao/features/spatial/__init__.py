"""Spatial augmenters for robot demonstration images."""

from . import random_crop
from .random_crop import RandomCrop

__all__ = ["random_crop", "RandomCrop"]
