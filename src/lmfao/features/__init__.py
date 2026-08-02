"""Feature modules are imported here so they register with LMFAO.

When adding a new feature:

1. Create a module in this package, for example `src/lmfao/features/lighting/shadow.py`.
2. Decorate the feature class with `@register_kernel_feature("lighting.shadow")`.
3. Import the class here so it is registered when `lmfao` is imported.
"""

__all__: list[str] = []
