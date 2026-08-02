from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict, Iterable, Sequence, Type

from lmfao.base import Augmenter


@dataclass(frozen=True)
class AugmenterInfo:
    name: str
    cls: Type[Augmenter]
    tags: tuple[str, ...]
    description: str


class AugmenterRegistry:
    """Central registry for feature modules."""

    def __init__(self) -> None:
        self._augmenters: Dict[str, AugmenterInfo] = {}

    def register(
        self,
        name: str,
        augmenter_cls: Type[Augmenter],
        tags: Sequence[str] = (),
        description: str = "",
    ) -> None:
        if not name:
            raise ValueError("augmenter name cannot be empty")
        if not issubclass(augmenter_cls, Augmenter):
            raise TypeError("augmenter_cls must inherit from Augmenter")
        if name in self._augmenters:
            raise ValueError(f"augmenter already registered: {name}")
        self._augmenters[name] = AugmenterInfo(
            name=name,
            cls=augmenter_cls,
            tags=tuple(tags),
            description=description or (augmenter_cls.__doc__ or "").strip(),
        )

    def get(self, name: str) -> Type[Augmenter]:
        try:
            return self._augmenters[name].cls
        except KeyError as exc:
            available = ", ".join(sorted(self._augmenters)) or "none"
            raise KeyError(f"unknown augmenter '{name}'. Available: {available}") from exc

    def info(self, name: str) -> AugmenterInfo:
        try:
            return self._augmenters[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._augmenters)) or "none"
            raise KeyError(f"unknown augmenter '{name}'. Available: {available}") from exc

    def build(self, name: str, **kwargs: Any) -> Augmenter:
        return self.get(name)(**kwargs)

    def list(self) -> list[str]:
        return sorted(self._augmenters)

    def list_info(self) -> list[AugmenterInfo]:
        return [self._augmenters[name] for name in self.list()]


registry = AugmenterRegistry()


def register_augmenter(name: str, tags: Sequence[str] = (), description: str = ""):
    """Decorator used by feature owners to expose their augmenter."""

    def decorator(augmenter_cls: Type[Augmenter]) -> Type[Augmenter]:
        registry.register(name, augmenter_cls, tags=tags, description=description)
        augmenter_cls.name = name
        return augmenter_cls

    return decorator


def get_augmenter(name: str) -> Type[Augmenter]:
    return registry.get(name)


def build_augmenter(name: str, **kwargs: Any) -> Augmenter:
    return registry.build(name, **kwargs)


def list_augmenters() -> list[str]:
    return registry.list()


def list_augmenter_info() -> list[AugmenterInfo]:
    return registry.list_info()


def build_many(configs: Iterable[dict[str, Any]]) -> list[Augmenter]:
    augmenters: list[Augmenter] = []
    for config in configs:
        item = dict(config)
        name = item.pop("name")
        params = item.pop("params", {})
        item.pop("probability", None)
        item.pop("enabled", None)
        if item:
            extra = ", ".join(sorted(item))
            raise ValueError(f"unknown config fields for '{name}': {extra}")
        augmenters.append(build_augmenter(name, **params))
    return augmenters


# Import built-ins once the registry helpers exist.
import_module("lmfao.features")
