"""Regression tests for the platform-sweep bug hunt.

Each test pins a bug found by the adversarial sweep so it cannot silently
return. Grouped by subsystem, not by finder.
"""

import numpy as np
import pytest

from lmfao.datasets import Episode
from lmfao.datasets.poses import assume_camera_track
from lmfao.features.lighting import BrightnessScale, ColorTemperatureShift, ContrastScale
from lmfao.features.noise import GaussianNoise, UniformNoise
from lmfao.features.occlusion.border_intrusion import BorderIntrusionOcclusion
from lmfao.features.occlusion.utils import sample_box
from lmfao.features.spatial.random_crop import RandomCrop
from lmfao.miniworld.config import CameraOffset, MiniWorldConfig
from lmfao.pipeline import AugmentationPipeline
from lmfao.program import generate_training_set

# --- feature param validation ------------------------------------------------

@pytest.mark.parametrize(
    "ctor, kwargs",
    [
        (BrightnessScale, {"factor": float("nan")}),
        (ContrastScale, {"factor": float("nan")}),
        (ColorTemperatureShift, {"shift": float("nan")}),
        (ColorTemperatureShift, {"intensity": float("nan")}),
        (GaussianNoise, {"sigma": float("nan")}),
        (UniformNoise, {"amplitude": float("nan")}),
        (BrightnessScale, {"factor": float("inf")}),
        (GaussianNoise, {"sigma": float("inf")}),
    ],
)
def test_non_finite_params_are_rejected(ctor, kwargs):
    with pytest.raises(ValueError):
        ctor(**kwargs)


# --- metadata integrity ------------------------------------------------------

def test_pipeline_does_not_mutate_caller_metadata():
    v = np.zeros((2, 8, 8, 3), np.uint8)
    _, m1 = AugmentationPipeline([BrightnessScale(factor=1.2)], seed=1)(v)
    recorded = dict(m1["augmentation_params"])
    AugmentationPipeline([BrightnessScale(factor=9.9)], seed=1)(v, m1)
    assert m1["augmentation_params"] == recorded


def test_augmenter_does_not_mutate_caller_metadata():
    v = np.zeros((2, 8, 8, 3), np.uint8)
    caller = {"augmentation_params": {"lighting.brightness": {"factor": 1.0}}}
    snapshot = {"lighting.brightness": {"factor": 1.0}}
    BrightnessScale(factor=2.0)(v, caller)
    assert caller["augmentation_params"] == snapshot


def test_repeated_augmenter_keeps_every_application_in_history():
    v = np.full((2, 8, 8, 3), 100, np.uint8)
    p = AugmentationPipeline([BrightnessScale(factor=0.5), BrightnessScale(factor=2.0)], seed=1)
    _, m = p(v)
    factors = [h["params"]["factor"] for h in m["augmentation_history"]]
    assert factors == [0.5, 2.0]


def test_random_crop_records_user_facing_pad_mode():
    rc = RandomCrop(pad=4, pad_mode="zero")
    assert rc.pad_mode == "zero"
    _, m = rc(np.zeros((2, 16, 16, 3), np.uint8), {}, np.random.default_rng(0))
    recorded = m["augmentation_params"]["spatial.random_crop"]["pad_mode"]
    assert recorded == "zero"
    # And the recorded value must rebuild the augmenter without error.
    AugmentationPipeline.from_config(
        [{"name": "spatial.random_crop", "params": {"pad": 4, "pad_mode": recorded}}]
    )


# --- occlusion ---------------------------------------------------------------

def test_rgba_occlusion_stays_opaque_black():
    v = np.full((1, 8, 8, 4), 255, np.uint8)
    out, meta = BorderIntrusionOcclusion(edges=("bottom",), max_fraction=0.5)(
        v, {}, np.random.default_rng(0)
    )
    r = meta["augmentation_params"]["occlusion.border_intrusion"]["regions"][0]
    region = out[0, r["top"]:r["top"] + r["height"], :, :]
    assert np.all(region[..., :3] == 0)  # black color
    assert np.all(region[..., 3] == 255)  # fully opaque


def test_sample_box_hits_target_area_on_non_square_frames():
    # 4 tall, 100 wide: height gets capped, width must grow to keep the area.
    box = sample_box(4, 100, (0.5, 0.5), (1.0, 1.0), np.random.default_rng(0))
    frac = box["height"] * box["width"] / (4 * 100)
    assert abs(frac - 0.5) < 0.1


# --- miniworld ---------------------------------------------------------------

def test_camera_offset_rejects_unknown_mapping_keys():
    with pytest.raises(ValueError, match="unknown camera offset"):
        CameraOffset.parse({"translatoin": [0.5, 0.0, 0.0]})


def test_camera_offset_pads_short_translation():
    off = CameraOffset.parse({"translation": [0.05, 0.0]})
    assert off.translation == (0.05, 0.0, 0.0)


def test_miniworld_config_rejects_bad_seed():
    with pytest.raises(ValueError):
        MiniWorldConfig.from_dict({"seed": -3})
    with pytest.raises(ValueError):
        MiniWorldConfig.from_dict({"seed": 1.5})


def test_generator_covers_distinct_source_offset_pairs(episode_factory):
    from lmfao.miniworld.generator import MiniWorldGenerator

    reals = [episode_factory(seed=s) for s in range(3)]
    cfg = MiniWorldConfig(enabled=True, n_synthetic=6, seed=7)
    out = MiniWorldGenerator(cfg).generate(reals)
    # 6 synthetic episodes should exercise all three default offsets, not the
    # single-offset-per-cycle the old lockstep produced on similar sources.
    offsets = {tuple(ep.metadata["miniworld"]["camera_offset"]["translation"]) for ep in out}
    assert len(offsets) == 3  # all three default offsets used


def test_generator_no_false_object_pose_edit(episode_factory):
    from lmfao.miniworld.generator import MiniWorldGenerator

    # puck_color matches nothing, so no puck is segmented and no edit happens.
    cfg = MiniWorldConfig(
        enabled=True, n_synthetic=1, seed=7, puck_color=(0.0, 1.0, 0.0),
        puck_color_tol=0.02, object_pose_region=((-0.05, -0.05), (0.05, 0.05)),
    )
    out = MiniWorldGenerator(cfg).generate([episode_factory(seed=0)])
    assert out[0].metadata["miniworld"]["object_pose_edit"] is None


def test_synthetic_metadata_not_aliased_to_source(episode_factory):
    from lmfao.miniworld.generator import MiniWorldGenerator

    source = episode_factory(seed=0)
    source.metadata["tags"] = ["real"]
    syn = MiniWorldGenerator(MiniWorldConfig(enabled=True, n_synthetic=1, seed=1)).generate(
        [source]
    )[0]
    syn.metadata["tags"].append("edit")
    assert source.metadata["tags"] == ["real"]


def test_renderer_survives_non_finite_gaussian():
    from lmfao.miniworld.camera import Camera, default_intrinsics, look_at
    from lmfao.miniworld.render import PointSplatRenderer
    from lmfao.miniworld.scene import GaussianCloud

    cam = Camera(default_intrinsics(32, 32), look_at(np.array([0.0, 0.0, -2.0]), np.zeros(3)), 32, 32)
    cloud = GaussianCloud(
        np.array([[0.0, 0.0, np.inf]]), np.ones((1, 3)), np.array([0.2]), np.array([1.0])
    )
    frame, holes = PointSplatRenderer().render(cloud, cam)  # must not raise
    assert frame.shape == (32, 32, 3)


# --- poses / episode ---------------------------------------------------------

def test_assume_camera_track_preserves_measured_poses():
    f = np.zeros((4, 16, 16, 3), np.uint8)
    poses = np.tile(np.eye(4), (4, 1, 1))
    poses[:, 0, 3] = [10, 11, 12, 13]
    out = assume_camera_track(Episode(frames=f, camera_poses=poses.copy()))
    assert np.allclose(out.camera_poses, poses)


def test_assume_camera_track_preserves_measured_intrinsics():
    f = np.zeros((4, 16, 16, 3), np.uint8)
    k = np.array([[123.0, 0, 7], [0, 456.0, 9], [0, 0, 1]])
    out = assume_camera_track(Episode(frames=f, intrinsics=k.copy()))
    assert np.allclose(out.intrinsics, k)


@pytest.mark.parametrize("bad_fps", [0.0, -30.0, float("nan"), float("inf")])
def test_episode_rejects_bad_fps(bad_fps):
    with pytest.raises(ValueError):
        Episode(frames=np.zeros((4, 16, 16, 3), np.uint8), fps=bad_fps)


# --- program -----------------------------------------------------------------

def test_adjacent_base_seeds_produce_independent_seasoning():
    f = np.random.default_rng(0).integers(0, 255, (4, 16, 16, 3), dtype=np.uint8)
    eps = [Episode(frames=f.copy()) for _ in range(3)]
    cfg = {"pipeline": [{"name": "noise.gaussian", "params": {"sigma": 0.2}}]}
    a = generate_training_set(eps, cfg, seed=100)
    b = generate_training_set(eps, cfg, seed=101)
    # seed+index would have made a.episodes[1] == b.episodes[0].
    assert not np.array_equal(a.episodes[1].frames, b.episodes[0].frames)


def test_generate_training_set_rejects_unknown_toplevel_key():
    eps = [Episode(frames=np.zeros((2, 8, 8, 3), np.uint8))]
    with pytest.raises(ValueError, match="unknown top-level"):
        generate_training_set(eps, {"minworld": {}}, seed=1)
