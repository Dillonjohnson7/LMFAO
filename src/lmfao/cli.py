"""``lmfao-augment``: apply an augmentation job to a whole LeRobot dataset.

This is the reliable, full-scale generator the browser demo delegates to. It
reads a LeRobot v3.0 dataset, and for every source episode and every variant in
the job config it augments the selected camera videos with the real
:class:`lmfao.pipeline.AugmentationPipeline`, copies the robot trajectory
(``state``/``actions``/``timestamps``) through unchanged, and writes a new v3.0
dataset. Output episodes = source episodes x variants.

Job config (``--config job.json``), the same schema the web demo exports::

    {
      "lmfao_job": 1,
      "seed": 42,
      "cameras": ["observation.images.wrist"],
      "variants": [
        {"id": "brightness-up", "suffix": "brightness_up",
         "pipeline": [{"name": "lighting.brightness", "params": {"factor": 1.35}, "probability": 1.0}]}
      ]
    }

``variants[].pipeline`` is exactly what ``AugmentationPipeline.from_config`` takes,
so a variant previewed in the browser reproduces here.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from lmfao.pipeline import AugmentationPipeline


def _seed_for(base: int, ep_i: int, var_i: int, cam_i: int) -> int:
    # Deterministic and distinct per (episode, variant, camera) so cameras are
    # augmented independently but the whole run is reproducible from --seed.
    return (base + ep_i * 1_000_003 + var_i * 10_007 + cam_i * 101) % (2**31)


def run(
    input_dir: str | Path,
    output_dir: str | Path,
    job: dict,
    *,
    seed: int,
    cameras: list[str] | None,
    limit: int | None,
) -> dict:
    # Imported here so `--help` works without the [datasets] extra installed.
    from lmfao.datasets.lerobot_reader import LeRobotReader
    from lmfao.datasets.lerobot_writer import LeRobotWriter

    reader = LeRobotReader(input_dir)
    cams = cameras or job.get("cameras") or reader.camera_keys
    missing = [c for c in cams if c not in reader.camera_keys]
    if missing:
        raise SystemExit(f"cameras not in dataset {reader.camera_keys}: {missing}")

    variants = job.get("variants") or []
    if not variants:
        raise SystemExit("job config has no variants")

    features = {k: v for k, v in reader.features.items() if v.get("dtype") != "video" or k in cams}
    writer = LeRobotWriter(
        output_dir,
        features=features,
        fps=reader.fps,
        robot_type=reader.info.get("robot_type", ""),
    )

    n_source = len(reader) if limit is None else min(limit, len(reader))
    written = 0
    for ep_i in range(n_source):
        ep = reader.read_episode(ep_i, cameras=cams)
        for var_i, variant in enumerate(variants):
            aug_frames = {}
            for cam_i, cam in enumerate(cams):
                pipe = AugmentationPipeline.from_config(
                    variant["pipeline"], seed=_seed_for(seed, ep_i, var_i, cam_i)
                )
                out, _ = pipe(ep.frames[cam])
                aug_frames[cam] = out
            writer.add_episode(
                aug_frames,
                state=ep.state,
                actions=ep.actions,
                timestamps=ep.timestamps,
                task=ep.task,
            )
            written += 1
        print(f"  episode {ep_i + 1}/{n_source} -> {len(variants)} variant(s)", file=sys.stderr)

    writer.close()
    return {"source_episodes": n_source, "variants": len(variants), "written_episodes": written}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lmfao-augment", description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="source LeRobot dataset directory")
    parser.add_argument("--output", required=True, help="output dataset directory (created)")
    parser.add_argument("--config", required=True, help="augmentation job JSON")
    parser.add_argument("--seed", type=int, default=None, help="override the job's base seed")
    parser.add_argument("--cameras", nargs="*", default=None, help="override which cameras to augment")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N source episodes")
    args = parser.parse_args(argv)

    job = json.loads(Path(args.config).read_text())
    seed = args.seed if args.seed is not None else int(job.get("seed", 0))

    summary = run(
        args.input,
        args.output,
        job,
        seed=seed,
        cameras=args.cameras,
        limit=args.limit,
    )
    print(
        f"wrote {summary['written_episodes']} episodes "
        f"({summary['source_episodes']} source x {summary['variants']} variants) to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
