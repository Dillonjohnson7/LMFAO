from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict, Iterable, Sequence, Type

from lmfao.base import AugmentationFeature


@dataclass(frozen=True)
class FeatureInfo:
    name: str
    cls: Type[AugmentationFeature]
    tags: tuple[str, ...]
    backends: tuple[str, ...]
    description: str


class FeatureRegistry:
    """Central registry for augmentation feature modules."""

    def __init__(self) -> None:
        self._features: Dict[str, FeatureInfo] = {}

    def register(
        self,
        name: str,
        feature_cls: Type[AugmentationFeature],
        tags: Sequence[str] = (),
        backends: Sequence[str] = (),
        description: str = "",
    ) -> None:
        if not name:
            raise ValueError("feature name cannot be empty")
        if not issubclass(feature_cls, AugmentationFeature):
            raise TypeError("feature_cls must inherit from AugmentationFeature")
        if name in self._features:
            raise ValueError(f"feature already registered: {name}")

        self._features[name] = FeatureInfo(
            name=name,
            cls=feature_cls,
            tags=tuple(tags),
            backends=tuple(backends),
            description=description or (feature_cls.__doc__ or "").strip(),
        )
        feature_cls.name = name
        feature_cls.supported_backends = tuple(backends)

    def get(self, name: str) -> Type[AugmentationFeature]:
        try:
            return self._features[name].cls
        except KeyError as exc:
            available = ", ".join(sorted(self._features)) or "none"
            raise KeyError(f"unknown feature '{name}'. Available: {available}") from exc

    def info(self, name: str) -> FeatureInfo:
        try:
            return self._features[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._features)) or "none"
            raise KeyError(f"unknown feature '{name}'. Available: {available}") from exc

    def build(self, name: str, **kwargs: Any) -> AugmentationFeature:
        return self.get(name)(**kwargs)

    def list(self) -> list[str]:
        return sorted(self._features)

    def list_info(self) -> list[FeatureInfo]:
        return [self._features[name] for name in self.list()]


registry = FeatureRegistry()


def register_feature(
    name: str,
    tags: Sequence[str] = (),
    backends: Sequence[str] = (),
    description: str = "",
):
    """Decorator used by feature owners to expose their augmentation feature."""

    def decorator(feature_cls: Type[AugmentationFeature]) -> Type[AugmentationFeature]:
        registry.register(name, feature_cls, tags=tags, backends=backends, description=description)
        return feature_cls

    return decorator


def get_feature(name: str) -> Type[AugmentationFeature]:
    return registry.get(name)


def build_feature(name: str, **kwargs: Any) -> AugmentationFeature:
    return registry.build(name, **kwargs)


def list_features() -> list[str]:
    return registry.list()


def list_feature_info() -> list[FeatureInfo]:
    return registry.list_info()


def build_many(configs: Iterable[dict[str, Any]]) -> list[AugmentationFeature]:
    features: list[AugmentationFeature] = []
    for config in configs:
        item = dict(config)
        name = item.pop("name")
        params = item.pop("params", {})
        item.pop("probability", None)
        item.pop("enabled", None)
        if item:
            extra = ", ".join(sorted(item))
            raise ValueError(f"unknown config fields for '{name}': {extra}")
        features.append(build_feature(name, **params))
    return features


# Import feature modules once the registry helpers exist.
import_module("lmfao.features")
