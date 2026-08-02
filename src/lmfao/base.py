from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, MutableMapping, Optional, Protocol

Video = Any
Metadata = MutableMapping[str, Any]


class AugmentationRuntime(Protocol):
    """Backend adapter responsible for executing augmentation operations.

    The core LMFAO package does not own tensors, dataframes, CUDA buffers,
    streams, or kernels. Runtime implementations own those details so the same
    feature can run through Torch/TorchVision, Pandas-backed dataset code, CUDA,
    Triton, or another backend.
    """

    backend: str

    def execute(
        self,
        operation_name: str,
        video: Video,
        params: Mapping[str, Any],
        metadata: Metadata,
        stream: Optional[Any] = None,
    ) -> Video:
        """Execute one named augmentation operation and return the video handle."""


class AugmentationFeature(ABC):
    """Base class for backend-agnostic video augmentation features.

    Feature classes should stay thin: validate/configure params, call the
    runtime, and record metadata. Heavy tensor, dataframe, or kernel work belongs
    in a runtime backend.
    """

    name: str
    supported_backends: tuple[str, ...] = ()
    supports_concurrent: bool = True

    def __call__(
        self,
        video: Video,
        runtime: AugmentationRuntime,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> tuple[Video, Metadata]:
        run_metadata: Metadata = dict(metadata or {})
        if self.supported_backends and runtime.backend not in self.supported_backends:
            supported = ", ".join(self.supported_backends)
            raise ValueError(f"{self.name} does not support backend '{runtime.backend}'. Supported: {supported}")
        return self.apply(video, runtime, run_metadata)

    @abstractmethod
    def apply(
        self,
        video: Video,
        runtime: AugmentationRuntime,
        metadata: Metadata,
    ) -> tuple[Video, Metadata]:
        """Execute this feature through the selected runtime backend."""
