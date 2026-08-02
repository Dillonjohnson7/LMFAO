"""Feature modules are imported here so they register with LMFAO.

When adding a new feature:

1. Create a module in this package, for example `src/lmfao/features/lighting.py`.
2. Decorate the augmenter class with `@register_augmenter("lighting")`.
3. Import the class here so it is registered when `lmfao` is imported.
"""

from lmfao.features.lighting import BrightnessScale, ColorTemperatureShift, ContrastScale
from lmfao.features.spatial import RandomCrop

__all__: list[str] = ["BrightnessScale", "ColorTemperatureShift", "ContrastScale", "RandomCrop"]
