#!/usr/bin/env python3
"""Re-encode dataset videos for fast random frame access."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av


def _already_low_gop(path: Path) -> bool:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = []
        for frame in container.decode(stream):
            frames.append(frame)
            if len(frames) == 3:
                break
    return len(frames) == 3 and frames[0].key_frame and frames[2].key_frame


def _reencode(path_string: str) -> tuple[str, str, int, int]:
    path = Path(path_string)
    if _already_low_gop(path):
        size = path.stat().st_size
        return str(path), "skipped", size, size

    temporary = path.with_name(f".{path.stem}.low-gop.tmp.mp4")
    temporary.unlink(missing_ok=True)
    old_size = path.stat().st_size

    try:
        with av.open(str(path)) as source, av.open(
            str(temporary), "w", format="mp4"
        ) as target:
            source_stream = source.streams.video[0]
            target_stream = target.add_stream(
                "libx264",
                rate=source_stream.average_rate or 30,
            )
            target_stream.width = source_stream.width
            target_stream.height = source_stream.height
            target_stream.pix_fmt = "yuv420p"
            target_stream.codec_context.gop_size = 2
            target_stream.codec_context.thread_count = 2
            target_stream.options = {
                "preset": "veryfast",
                "crf": "18",
                "keyint": "2",
                "min-keyint": "2",
                "scenecut": "0",
            }

            frame_count = 0
            for frame in source.decode(source_stream):
                frame_count += 1
                for packet in target_stream.encode(frame):
                    target.mux(packet)
            for packet in target_stream.encode():
                target.mux(packet)

        with av.open(str(temporary)) as check:
            check_stream = check.streams.video[0]
            packets = sum(
                packet.size > 0 for packet in check.demux(check_stream)
            )
        if packets != frame_count or not _already_low_gop(temporary):
            raise RuntimeError(
                f"validation failed: frames={frame_count}, packets={packets}"
            )

        new_size = temporary.stat().st_size
        os.replace(temporary, path)
        return str(path), "encoded", old_size, new_size
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    paths = sorted(
        path
        for path in args.root.rglob("*.mp4")
        if not path.name.endswith(".low-gop.tmp.mp4")
    )
    if not paths:
        raise SystemExit(f"no MP4 files found under {args.root}")

    print(f"START files={len(paths)} workers={args.workers}", flush=True)
    encoded = skipped = old_bytes = new_bytes = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_reencode, str(path)): path for path in paths}
        for completed, future in enumerate(as_completed(futures), 1):
            path, status, before, after = future.result()
            encoded += status == "encoded"
            skipped += status == "skipped"
            old_bytes += before
            new_bytes += after
            print(
                f"PROGRESS {completed}/{len(paths)} {status} "
                f"{before}->{after} {path}",
                flush=True,
            )

    print(
        f"DONE encoded={encoded} skipped={skipped} "
        f"bytes={old_bytes}->{new_bytes}",
        flush=True,
    )


if __name__ == "__main__":
    main()
