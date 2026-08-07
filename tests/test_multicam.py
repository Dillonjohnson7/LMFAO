"""Hammer dual-cam preserve-all: front + wrist must survive augment I/O."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("av")

from lmfao.datasets import Episode, read_lerobot_dataset, write_lerobot_dataset
from lmfao.datasets.lerobot import LeRobotStreamingWriter
from lmfao.program import augment_episodes
from lmfao.replay import replay_augmentation_history

FRONT = "observation.images.front"
WRIST = "observation.images.wrist"


def _dual(frames=8, seed=0):
    rng = np.random.default_rng(seed)
    front = rng.integers(0, 256, (frames, 48, 64, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, (frames, 72, 96, 3), dtype=np.uint8)
    state = np.zeros((frames, 6), dtype=float)
    return Episode(
        frames=front,
        state=state,
        actions=state.copy(),
        fps=30.0,
        task="pick",
        metadata={"video_key": FRONT, "video_keys": [FRONT, WRIST]},
        extra_videos={WRIST: wrist},
    )


def test_episode_rejects_mismatched_extra_length():
    ep = _dual()
    with pytest.raises(ValueError, match="frames"):
        Episode(
            frames=ep.frames,
            metadata={"video_key": FRONT},
            extra_videos={WRIST: ep.extra_videos[WRIST][:4]},
        )


def test_write_read_preserves_both_cameras(tmp_path):
    write_lerobot_dataset([_dual(seed=1), _dual(seed=2)], tmp_path)
    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    assert FRONT in info["features"] and WRIST in info["features"]
    assert info["features"][FRONT]["shape"] == [48, 64, 3]
    assert info["features"][WRIST]["shape"] == [72, 96, 3]
    assert (tmp_path / "videos" / FRONT / "chunk-000" / "file-000.mp4").exists()
    assert (tmp_path / "videos" / WRIST / "chunk-000" / "file-000.mp4").exists()

    back = read_lerobot_dataset(tmp_path)
    assert len(back) == 2
    for ep in back:
        assert ep.metadata["video_key"] == FRONT
        assert WRIST in ep.extra_videos
        assert ep.frames.shape[1:] == (48, 64, 3)
        assert ep.extra_videos[WRIST].shape[1:] == (72, 96, 3)


def test_single_video_key_drops_extras_on_read(tmp_path):
    write_lerobot_dataset([_dual()], tmp_path)
    back = read_lerobot_dataset(tmp_path, video_key=WRIST)
    assert back[0].metadata["video_key"] == WRIST
    assert back[0].extra_videos == {}
    assert back[0].frames.shape[1:] == (72, 96, 3)


def test_streaming_writer_writes_both_cameras(tmp_path):
    w = LeRobotStreamingWriter(
        tmp_path, video_keys=[FRONT, WRIST], fps=30.0, state_dim=6, action_dim=6
    )
    w.add_episode(_dual(seed=3))
    w.add_episode(_dual(seed=4))
    w.close()
    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    assert set(_ for _ in info["features"] if _.startswith("observation.images.")) == {
        FRONT,
        WRIST,
    }
    for ei in (0, 1):
        assert (tmp_path / "videos" / FRONT / "chunk-000" / f"file-{ei:03d}.mp4").exists()
        assert (tmp_path / "videos" / WRIST / "chunk-000" / f"file-{ei:03d}.mp4").exists()
    back = read_lerobot_dataset(tmp_path)
    assert all(WRIST in ep.extra_videos for ep in back)


def test_augment_preserves_extra_camera_and_shared_brightness():
    pipe = [{"name": "lighting.brightness", "params": {"factor": 0.5}, "probability": 1.0}]
    src = _dual(seed=9)
    before_wrist = src.extra_videos[WRIST].copy()
    out = augment_episodes([src], pipe, variants=1, seed=1, include_original=True)
    assert len(out.episodes) == 2
    orig, aug = out.episodes
    assert not orig.metadata["augmented"]
    assert WRIST in orig.extra_videos
    assert np.array_equal(orig.extra_videos[WRIST], before_wrist)
    assert aug.metadata["augmented"]
    assert WRIST in aug.extra_videos
    # Primary darkened ~0.5; wrist should move the same direction.
    assert float(aug.frames.mean()) < float(src.frames.mean()) * 0.7
    assert float(aug.extra_videos[WRIST].mean()) < float(before_wrist.mean()) * 0.7
    assert np.array_equal(src.extra_videos[WRIST], before_wrist)  # source untouched


def test_replay_border_intrusion_adapts_to_geometry():
    from lmfao.pipeline import AugmentationPipeline

    primary = np.full((6, 40, 60, 3), 200, np.uint8)
    secondary = np.full((6, 80, 120, 3), 200, np.uint8)
    pipe = AugmentationPipeline.from_config(
        [{"name": "occlusion.border_intrusion", "params": {"max_fraction": 0.2}, "probability": 1.0}],
        seed=3,
    )
    out_p, meta = pipe(primary, {})
    hist = meta["augmentation_history"]
    out_s = replay_augmentation_history(secondary, hist, primary_shape=primary.shape)
    assert out_s.shape == secondary.shape
    # Both should have some occluded (black) pixels.
    assert (out_p == 0).any() and (out_s == 0).any()


def test_replay_both_noise_types_uses_recorded_parameter_names():
    from lmfao.pipeline import AugmentationPipeline

    primary = np.full((4, 24, 32, 3), 128, np.uint8)
    secondary = primary.copy()
    pipe = AugmentationPipeline.from_config(
        [
            {"name": "noise.gaussian", "params": {"sigma": 0.03}},
            {"name": "noise.uniform", "params": {"amplitude": 0.04}},
        ],
        seed=5,
    )
    _, meta = pipe(primary, {})
    history = meta["augmentation_history"]
    assert history[0]["params"]["sigma"] == 0.03
    assert history[1]["params"]["amplitude"] == 0.04

    replayed = replay_augmentation_history(secondary, history, primary_shape=primary.shape)
    assert replayed.shape == secondary.shape
    assert not np.array_equal(replayed, secondary)


def test_cli_augment_multicam_e2e(tmp_path):
    from lmfao.cli import main

    stock = tmp_path / "stock"
    out = tmp_path / "aug"
    write_lerobot_dataset([_dual(seed=1), _dual(seed=2)], stock)
    cfg = tmp_path / "pipe.json"
    cfg.write_text(
        json.dumps(
            [
                {"name": "lighting.brightness", "params": {"min_factor": 0.8, "max_factor": 1.2}},
                {"name": "noise.gaussian", "params": {"sigma": 0.01}},
            ]
        )
    )
    rc = main(
        [
            "augment",
            "--input",
            str(stock),
            "--output",
            str(out),
            "--config",
            str(cfg),
            "--variants",
            "1",
            "--include-original",
            "--seed",
            "7",
            "--overwrite",
        ]
    )
    assert rc == 0
    info = json.loads((out / "meta" / "info.json").read_text())
    assert FRONT in info["features"] and WRIST in info["features"]
    assert info["total_episodes"] == 4  # 2 originals + 2 variants
    back = read_lerobot_dataset(out)
    assert all(WRIST in ep.extra_videos for ep in back)
    assert (out / "videos" / WRIST).exists()
    # No silent drop: video file count matches episodes for each cam.
    assert len(list((out / "videos" / FRONT).rglob("*.mp4"))) == 4
    assert len(list((out / "videos" / WRIST).rglob("*.mp4"))) == 4
