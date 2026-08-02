"""Feature modules are imported here so they register with LMFAO.

When adding a new feature:

1. Create a module, for example `src/lmfao/features/occlusion/sequence_box.py`.
2. Decorate the augmenter class with `@register_augmenter("occlusion.sequence_box")`.
3. Import the package here so it is registered when `lmfao` is imported.
"""

from . import lighting, noise, occlusion, spatial

__all__ = ["lighting", "noise", "occlusion", "spatial"]
