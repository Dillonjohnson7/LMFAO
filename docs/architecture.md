# Architecture

LMFAO has four lightweight layers.

```text
dataset -> pipeline -> feature -> backend runtime -> augmented dataset
```

## Dataset Layer

The dataset layer is responsible for reading and writing robot demonstration
episodes. For ACT-style policies, each episode contains observation sequences
and aligned action chunks.

Image-only transforms must preserve:

- episode ordering
- frame/action alignment
- action values
- action chunk labels

## Pipeline Layer

The pipeline loads feature configs, applies probability gates, and calls each
feature in order. It should not know how pixels are modified.

## Feature Layer

Features define augmentation semantics and parameters. They do not contain heavy
backend-specific math.

Example:

```text
occlusion.random_box = mask a rectangle in RGB observations
```

## Backend Layer

Backends execute operations for a concrete compute/data stack.

Examples:

- `torch_cpu`: Torch tensor ops on CPU
- `torchvision_cpu`: TorchVision image/video transforms
- `pandas_manifest`: dataset manifest expansion or bookkeeping
- `torch_cuda`: Torch tensor ops on CUDA tensors
- `cuda`: custom CUDA kernels
- `triton`: Triton kernels

The backend owns imports, device placement, memory layout, stream handling, and
any concurrency guarantees.

## Why This Split

This keeps LMFAO light while letting the team move in parallel:

- feature owners define augmentation behavior
- backend owners optimize execution
- configs stay stable across CPU and GPU implementations
- ACT labels remain valid for image-only transforms
