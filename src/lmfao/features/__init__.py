"""Feature modules are imported here so they register with LMFAO.

When adding a new feature:

1. Create a module, for example `src/lmfao/features/occlusion/random_box.py`.
2. Decorate the feature class with `@register_feature("occlusion.random_box")`.
3. Import the module here so it is registered when `lmfao` is imported.
"""

__all__: list[str] = []
