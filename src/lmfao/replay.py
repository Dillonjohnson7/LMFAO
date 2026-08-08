"""Replay a recorded augmentation history onto another camera stream.

Primary stream is seasoned first (samples params into ``augmentation_history``).
Secondary streams must get the *same* seasoning decision — same lighting factors,
same occlusion fractions, same crop pad — adapted to their own geometry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from lmfao.base import Video
from lmfao.registry import build_augmenter

# Noises drawn per pixel, and the keyword each one records its strength under.
_SAMPLED_NOISE = {
    "noise.gaussian": "sigma",
    "noise.uniform": "amplitude",
    "noise.shot": "strength",
}


def replay_augmentation_history(
    video: Video,
    history: Sequence[Mapping[str, Any]],
    *,
    primary_shape: tuple[int, int, int, int] | None = None,
    rng: np.random.Generator | None = None,
) -> Video:
    """Apply each history step to ``video`` using the recorded parameters.

    ``primary_shape`` is ``(F, H, W, C)`` of the stream the history was sampled
    on; used to scale absolute pixel shifts when geometries differ.
    """
    out = video
    run_rng = rng if rng is not None else np.random.default_rng(0)
    src_hw = None
    if primary_shape is not None and len(primary_shape) >= 3:
        src_hw = (int(primary_shape[1]), int(primary_shape[2]))
    for step in history:
        name = str(step["name"])
        params = dict(step.get("params") or {})
        out = _replay_one(out, name, params, run_rng, src_hw=src_hw)
    return out


def _replay_one(
    video: Video,
    name: str,
    params: dict[str, Any],
    rng: np.random.Generator,
    *,
    src_hw: tuple[int, int] | None,
) -> Video:
    if name == "lighting.brightness":
        aug = build_augmenter(name, factor=float(params["factor"]))
        return aug(video, {}, rng)[0]

    if name == "lighting.contrast":
        # pivot is absolute mid-gray of the source clip; recompute on this stream
        # by leaving factor fixed (pivot is derived inside apply from the video).
        aug = build_augmenter(name, factor=float(params["factor"]))
        return aug(video, {}, rng)[0]

    if name == "lighting.color_temperature":
        aug = build_augmenter(
            name,
            shift=float(params["shift"]),
            intensity=float(params.get("intensity", 0.35)),
        )
        return aug(video, {}, rng)[0]

    if name in _SAMPLED_NOISE:
        # Same strength; independent realization (correct for multi-sensor noise).
        # Each records its strength under its own keyword, so look it up by name.
        key = _SAMPLED_NOISE[name]
        kwargs: dict[str, Any] = {key: float(params[key])}
        if "device" in params and params["device"] not in (None, "auto"):
            kwargs["device"] = params["device"]
        aug = build_augmenter(name, **kwargs)
        return aug(video, {}, rng)[0]

    if name == "noise.compression":
        # Deterministic given quality, and quality is geometry-independent.
        aug = build_augmenter(name, quality=float(params["quality"]))
        return aug(video, {}, rng)[0]

    if name == "noise.blur":
        # Radius is in pixels, so a wider stream needs a proportionally wider one.
        radius = float(params["radius"])
        if src_hw is not None and src_hw[1] > 0:
            radius *= video.shape[2] / src_hw[1]
        aug = build_augmenter(name, radius=max(radius, 1e-3))
        return aug(video, {}, rng)[0]

    if name == "spatial.random_crop":
        return _replay_random_crop(video, params, src_hw=src_hw, rng=rng)

    if name == "occlusion.border_intrusion":
        return _replay_border_intrusion(video, params)

    if name == "occlusion.sequence_box":
        return _replay_sequence_box(video, params, src_hw=src_hw)

    if name == "occlusion.moving_box":
        return _replay_moving_box(video, params, src_hw=src_hw)

    raise ValueError(
        f"cannot replay augmenter {name!r} onto a secondary camera; "
        "add a replay handler or pass --video-key to augment a single stream"
    )


def _replay_random_crop(
    video: Video,
    params: dict[str, Any],
    *,
    src_hw: tuple[int, int] | None,
    rng: np.random.Generator,
) -> Video:
    pad = int(params["pad"])
    pad_mode = str(params.get("pad_mode", "reflect"))
    shift_x = np.asarray(params["shift_x"], dtype=int)
    shift_y = np.asarray(params["shift_y"], dtype=int)
    f, h, w, _ = video.shape
    if shift_x.shape[0] != f or shift_y.shape[0] != f:
        raise ValueError(
            f"spatial.random_crop history has {shift_x.shape[0]} shifts but video has {f} frames"
        )
    if src_hw is not None:
        src_h, src_w = src_hw
        if (src_h, src_w) != (h, w) and src_h > 0 and src_w > 0:
            # Scale pixel shifts into the destination geometry, then clamp to pad.
            shift_x = np.clip(np.rint(shift_x * (w / src_w)).astype(int), -pad, pad)
            shift_y = np.clip(np.rint(shift_y * (h / src_h)).astype(int), -pad, pad)
    # pad must fit this geometry
    if pad >= h or pad >= w:
        # fall back: shrink pad so the crop is still defined
        pad = max(1, min(h, w) // 4)
        shift_x = np.clip(shift_x, -pad, pad)
        shift_y = np.clip(shift_y, -pad, pad)
    aug = build_augmenter(
        "spatial.random_crop",
        pad=pad,
        pad_mode=pad_mode,
        shift_x=shift_x.tolist(),
        shift_y=shift_y.tolist(),
    )
    return aug(video, {}, rng)[0]


def _replay_border_intrusion(video: Video, params: dict[str, Any]) -> Video:
    """Rebuild from edge+fraction (geometry-independent), not absolute pixels."""
    from lmfao.features.occlusion.border_intrusion import _border_region, _fill_border

    out = video.copy()
    fill = params.get("fill", "black")
    temporal = params.get("temporal_mode", "constant")
    regions = list(params.get("regions") or [])
    if not regions:
        return out
    _, height, width, _ = out.shape
    if temporal == "constant":
        r0 = regions[0]
        region = _border_region(str(r0["edge"]), float(r0["fraction"]), height, width)
        _fill_border(out, region, fill)
        return out
    for r0 in regions:
        frame = r0.get("frame")
        if frame is None:
            continue
        region = _border_region(str(r0["edge"]), float(r0["fraction"]), height, width)
        _fill_border(out[int(frame) : int(frame) + 1], region, fill)
    return out


def _scale_box(box: Mapping[str, Any], src_hw: tuple[int, int] | None, h: int, w: int) -> dict[str, int]:
    """Map a recorded box to a new geometry via normalized position/size."""
    bh = max(1, int(box["height"]))
    bw = max(1, int(box["width"]))
    top = int(box["top"])
    left = int(box["left"])
    if src_hw is None:
        src_h, src_w = h, w
    else:
        src_h, src_w = src_hw
    # Normalize then re-materialize.
    top_n = top / max(src_h, 1)
    left_n = left / max(src_w, 1)
    h_n = bh / max(src_h, 1)
    w_n = bw / max(src_w, 1)
    height = max(1, min(h, int(round(h_n * h))))
    width = max(1, min(w, int(round(w_n * w))))
    top_i = int(np.clip(round(top_n * h), 0, max(0, h - height)))
    left_i = int(np.clip(round(left_n * w), 0, max(0, w - width)))
    return {"top": top_i, "left": left_i, "height": height, "width": width}


def _replay_sequence_box(
    video: Video, params: dict[str, Any], *, src_hw: tuple[int, int] | None
) -> Video:
    from lmfao.features.occlusion.utils import fill_region

    out = video.copy()
    _, height, width, _ = out.shape
    box = _scale_box(params["box"], src_hw, height, width)
    fill = params.get("fill", "black")
    top, left = box["top"], box["left"]
    fill_region(out[:, top : top + box["height"], left : left + box["width"], :], fill)
    return out


def _replay_moving_box(
    video: Video, params: dict[str, Any], *, src_hw: tuple[int, int] | None
) -> Video:
    from lmfao.features.occlusion.moving_box import _position
    from lmfao.features.occlusion.utils import fill_region

    out = video.copy()
    frames, height, width, _ = out.shape
    box_rec = dict(params.get("box") or {})
    start = dict(params.get("start") or {"top": 0, "left": 0})
    # Attach start into box for _scale_box of size; start position scaled separately.
    sized = _scale_box(
        {
            "top": start.get("top", 0),
            "left": start.get("left", 0),
            "height": box_rec.get("height", 1),
            "width": box_rec.get("width", 1),
        },
        src_hw,
        height,
        width,
    )
    bh, bw = sized["height"], sized["width"]
    top0, left0 = sized["top"], sized["left"]
    velocity = params.get("velocity") or {"y": 0.0, "x": 0.0}
    fill = params.get("fill", "black")
    edge_bounce = bool(params.get("edge_bounce", True))
    src_h, src_w = src_hw if src_hw is not None else (height, width)
    vy = float(velocity.get("y", 0.0)) * (height / max(src_h, 1))
    vx = float(velocity.get("x", 0.0)) * (width / max(src_w, 1))

    for frame_index in range(frames):
        top = _position(top0, vy, frame_index, height - bh, edge_bounce)
        left = _position(left0, vx, frame_index, width - bw, edge_bounce)
        fill_region(out[frame_index, top : top + bh, left : left + bw, :], fill)
    return out