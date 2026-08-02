from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, MutableMapping, Optional, Protocol, Sequence

Video = Any
Metadata = MutableMapping[str, Any]


class KernelRuntime(Protocol):
    """Runtime adapter responsible for launching GPU kernels.

    The core library does not own CUDA, streams, or buffers. A project-specific
    runtime can wrap CUDA C++, CuPy, PyTorch extensions, Triton, or another GPU
    backend while the registry and pipeline stay lightweight.
    """

    def launch_kernel(
        self,
        kernel_name: str,
        grid: Any,
        block: Any,
        args: Sequence[Any],
        stream: Optional[Any] = None,
    ) -> None:
        ...


class KernelFeature(ABC):
    """Base class for GPU-backed video augmentation features.

    Implementations should launch kernels against an existing GPU-resident
    video handle. They should avoid CPU frame loops, host copies, and heavy
    allocations in the hot path.
    """

    name: str
    supports_concurrent: bool = True

    def __call__(
        self,
        video: Video,
        runtime: KernelRuntime,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Metadata:
        run_metadata: Metadata = dict(metadata or {})
        return self.launch(video, runtime, run_metadata)

    @abstractmethod
    def launch(self, video: Video, runtime: KernelRuntime, metadata: Metadata) -> Metadata:
        """Launch one or more GPU kernels for this feature."""
