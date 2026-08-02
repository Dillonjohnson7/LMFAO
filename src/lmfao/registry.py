from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict, Iterable, Sequence, Type

from lmfao.base import KernelFeature


@dataclass(frozen=True)
class KernelFeatureInfo:
    name: str
    cls: Type[KernelFeature]
    tags: tuple[str, ...]
    description: str


class KernelFeatureRegistry:
    """Central registry for GPU kernel feature modules."""

    def __init__(self) -> None:
        self._features: Dict[str, KernelFeatureInfo] = {}

    def register(
        self,
        name: str,
        feature_cls: Type[KernelFeature],
        tags: Sequence[str] = (),
        description: str = "",
    ) -> None:
        if not name:
            raise ValueError("feature name cannot be empty")
        if not issubclass(feature_cls, KernelFeature):
            raise TypeError("feature_cls must inherit from KernelFeature")
        if name in self._features:
            raise ValueError(f"feature already registered: {name}")
        self._features[name] = KernelFeatureInfo(
            name=name,
            cls=feature_cls,
            tags=tuple(tags),
            description=description or (feature_cls.__doc__ or "").strip(),
        )

    def get(self, name: str) -> Type[KernelFeature]:
        try:
            return self._features[name].cls
        except KeyError as exc:
            available = ", ".join(sorted(self._features)) or "none"
            raise KeyError(f"unknown feature '{name}'. Available: {available}") from exc

    def info(self, name: str) -> KernelFeatureInfo:
        try:
            return self._features[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._features)) or "none"
            raise KeyError(f"unknown feature '{name}'. Available: {available}") from exc

    def build(self, name: str, **kwargs: Any) -> KernelFeature:
        return self.get(name)(**kwargs)

    def list(self) -> list[str]:
        return sorted(self._features)

    def list_info(self) -> list[KernelFeatureInfo]:
        return [self._features[name] for name in self.list()]


registry = KernelFeatureRegistry()


def register_kernel_feature(name: str, tags: Sequence[str] = (), description: str = ""):
    """Decorator used by feature owners to expose their kernel feature."""

    def decorator(feature_cls: Type[KernelFeature]) -> Type[KernelFeature]:
        registry.register(name, feature_cls, tags=tags, description=description)
        feature_cls.name = name
        return feature_cls

    return decorator


def get_kernel_feature(name: str) -> Type[KernelFeature]:
    return registry.get(name)


def build_kernel_feature(name: str, **kwargs: Any) -> KernelFeature:
    return registry.build(name, **kwargs)


def list_kernel_features() -> list[str]:
    return registry.list()


def list_kernel_feature_info() -> list[KernelFeatureInfo]:
    return registry.list_info()


def build_many(configs: Iterable[dict[str, Any]]) -> list[KernelFeature]:
    features: list[KernelFeature] = []
    for config in configs:
        item = dict(config)
        name = item.pop("name")
        params = item.pop("params", {})
        item.pop("probability", None)
        item.pop("enabled", None)
        if item:
            extra = ", ".join(sorted(item))
            raise ValueError(f"unknown config fields for '{name}': {extra}")
        features.append(build_kernel_feature(name, **params))
    return features


# Import built-ins once the registry helpers exist.
import_module("lmfao.features")
