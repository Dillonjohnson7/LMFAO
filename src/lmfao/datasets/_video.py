"""Video encode/decode helpers for the LeRobot dataset layer.

Kept behind the ``[datasets]`` extra: PyAV is only imported here, so the core
``lmfao`` install stays numpy-only. Encoding targets H.264 in an mp4 container
(``yuv420p``), which every LeRobot loader and browser can decode; decoding is
frame accurate against the packed videos LeRobot v3.0 produces, where several
episodes share one file and are addressed by a ``[from_timestamp, to_timestamp)``
window.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np

# AV_TIME_BASE: PyAV's Container.seek offset unit when no stream is given.
_AV_TIME_BASE = 1_000_000


def _lazy_av():
    try:
        import av  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "the LeRobot dataset layer needs PyAV; install it with `pip install \"lmfao[datasets]\"`"
        ) from exc
    return av


def encode_mp4(
    path: str | Path,
    frames: np.ndarray,
    fps: float,
    *,
    codec: str = "libx264",
    pix_fmt: str = "yuv420p",
    crf: int = 23,
    preset: str | None = "veryfast",
    bitrate: int | None = None,
) -> None:
    """Encode ``(F, H, W, 3)`` uint8 RGB frames to an mp4 at ``path``.

    ``preset`` tunes the libx264 speed/size tradeoff (``ultrafast`` .. ``slow``);
    ``veryfast`` is ~2.4x faster than the x264 default ``medium`` and produces
    *smaller* files at the same CRF, so it is the default. Hardware encoders
    (e.g. ``h264_videotoolbox``) ignore CRF/preset and take a target ``bitrate``
    (bits/s) instead.
    """
    av = _lazy_av()
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be (F, H, W, 3) RGB, got shape {frames.shape}")
    if frames.shape[0] == 0:
        raise ValueError("cannot encode a video with 0 frames")
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _f, height, width, _c = frames.shape
    if height % 2 or width % 2:
        raise ValueError(
            f"H.264/yuv420p needs even frame dimensions, got {width}x{height}; "
            "crop or pad to even width/height before writing"
        )

    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream(codec, rate=Fraction(fps).limit_denominator(1000))
        stream.width = width
        stream.height = height
        stream.pix_fmt = pix_fmt
        if bitrate is not None:
            stream.bit_rate = int(bitrate)
        options: dict[str, str] = {}
        if codec.startswith("libx264") or codec.startswith("libx265"):
            options["crf"] = str(crf)
            if preset:
                options["preset"] = preset
        if options:
            stream.options = options
        for frame in frames:
            vframe = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
            for packet in stream.encode(vframe):
                container.mux(packet)
        for packet in stream.encode():  # flush
            container.mux(packet)
    finally:
        container.close()


def decode_window(
    path: str | Path,
    from_timestamp: float,
    to_timestamp: float,
    *,
    expected: int | None = None,
) -> np.ndarray:
    """Decode frames whose presentation time is in ``[from_timestamp, to_timestamp)``.

    Returns an ``(F, H, W, 3)`` uint8 array. ``expected`` caps the count and lets
    the decoder stop early once an episode's worth of frames has been collected.
    """
    av = _lazy_av()
    path = Path(path)
    container = av.open(str(path))
    out: list[np.ndarray] = []
    try:
        stream = container.streams.video[0]
        if from_timestamp > 0:
            # Seek to the keyframe at or before the window start, then filter.
            container.seek(int(max(from_timestamp, 0.0) * _AV_TIME_BASE), backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None:
                t = None
            else:
                t = float(frame.pts * stream.time_base)
            if t is not None and t < from_timestamp - 1e-4:
                continue
            if t is not None and t >= to_timestamp - 1e-4:
                break
            out.append(frame.to_ndarray(format="rgb24"))
            if expected is not None and len(out) >= expected:
                break
    finally:
        container.close()

    if not out:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)
    return np.stack(out).astype(np.uint8, copy=False)
