import numpy as np
import pytest
from conftest import PUCK_COLOR

from lmfao.datasets import Episode
from lmfao.miniworld import (
    Camera,
    FKBeaconTracker,
    GaussianCloud,
    MiniWorldConfig,
    MiniWorldGenerator,
    PointSplatReconstructor,
    PointSplatRenderer,
    SimpleInpainter,
    default_intrinsics,
    look_at,
)

# ----------------------------- camera ------------------------------------


def test_camera_project_backproject_roundtrip():
    k = default_intrinsics(64, 48, 55.0)
    pose = look_at(np.array([0.2, -0.4, 1.0]), np.array([0.0, 0.0, 0.0]))
    cam = Camera(k, pose, 64, 48)
    pts = np.array([[0.0, 0.0, 0.0], [0.1, -0.05, 0.02], [-0.1, 0.1, -0.03]])
    uv, depth = cam.project(pts)
    recovered = cam.backproject(uv, depth)
    np.testing.assert_allclose(recovered, pts, atol=1e-6)


def test_camera_offset_moves_center():
    k = default_intrinsics(32, 32)
    cam = Camera(k, look_at(np.array([0, 0, 1.0]), np.array([0, 0, 0.0])), 32, 32)
    moved = cam.offset(translation=(0.1, 0.0, 0.0))
    assert not np.allclose(cam.center, moved.center)
    assert np.isclose(np.linalg.norm(moved.center - cam.center), 0.1)


# --------------------------- gaussian cloud ------------------------------


def test_gaussiancloud_validates_and_clips():
    cloud = GaussianCloud(
        means=np.zeros((2, 3)),
        colors=np.array([[2.0, -1.0, 0.5], [0.1, 0.2, 0.3]]),
        scales=np.array([-1.0, 0.5]),
        opacities=np.array([2.0, 0.5]),
    )
    assert len(cloud) == 2
    assert cloud.colors.max() <= 1.0 and cloud.colors.min() >= 0.0
    assert (cloud.scales > 0).all()
    assert cloud.opacities.max() <= 1.0


def test_gaussiancloud_select_translate_concat():
    cloud = GaussianCloud(np.arange(6).reshape(2, 3).astype(float), np.ones((2, 3)), np.ones(2), np.ones(2))
    only_second = cloud.select(np.array([False, True]))
    assert len(only_second) == 1
    shifted = only_second.translated(np.array([1.0, 1.0, 1.0]))
    np.testing.assert_allclose(shifted.means, only_second.means + 1.0)
    combined = GaussianCloud.concat([cloud, shifted])
    assert len(combined) == 3
    assert len(GaussianCloud.concat([GaussianCloud.empty()])) == 0


# --------------------------- reconstruction ------------------------------


def test_reconstruct_requires_camera_geometry():
    ep = Episode(frames=np.zeros((3, 16, 16, 3), np.uint8))
    with pytest.raises(ValueError):
        PointSplatReconstructor().reconstruct(ep)


def test_reconstruct_produces_gaussians_and_segments_puck(episode_factory):
    ep = episode_factory(seed=1)
    recon = PointSplatReconstructor(pixel_stride=2, max_frames=4, puck_color=PUCK_COLOR, puck_color_tol=0.25)
    scene = recon.reconstruct(ep)
    assert scene.num_gaussians > 0
    assert len(scene.puck) > 0  # red puck segmented out of the static cloud
    assert scene.puck_home is not None
    assert scene.puck_home.shape == (3,)


# ------------------------------ rendering --------------------------------


def test_render_at_original_pose_has_few_holes(episode_factory):
    ep = episode_factory(seed=2)
    scene = PointSplatReconstructor(pixel_stride=1, max_frames=6).reconstruct(ep)
    renderer = PointSplatRenderer(background=(0.1, 0.1, 0.1))
    cam = Camera(ep.intrinsics, ep.camera_poses[0], ep.width, ep.height)
    frame, hole = renderer.render(scene.render_cloud(), cam)
    assert frame.shape == (ep.height, ep.width, 3)
    assert frame.dtype == np.uint8
    assert hole.mean() < 0.05  # re-rendering a real view fills almost everything


def test_novel_view_differs_from_original(episode_factory):
    ep = episode_factory(seed=3)
    scene = PointSplatReconstructor(pixel_stride=1, max_frames=6).reconstruct(ep)
    renderer = PointSplatRenderer()
    cam = Camera(ep.intrinsics, ep.camera_poses[0], ep.width, ep.height)
    base, _ = renderer.render(scene.render_cloud(), cam)
    novel, _ = renderer.render(scene.render_cloud(), cam.offset(translation=(0.15, 0.0, 0.0), yaw=0.1))
    assert np.abs(novel.astype(int) - base.astype(int)).mean() > 1.0


def test_empty_cloud_is_all_holes():
    renderer = PointSplatRenderer()
    cam = Camera(default_intrinsics(16, 16), look_at(np.array([0, 0, 1.0]), np.array([0, 0, 0.0])), 16, 16)
    frame, hole = renderer.render(GaussianCloud.empty(), cam)
    assert hole.all()
    assert frame.shape == (16, 16, 3)


# ------------------------------ inpainting -------------------------------


def test_inpaint_fills_all_holes():
    frame = np.zeros((10, 10, 3), np.uint8)
    frame[:, :5] = 200
    mask = np.zeros((10, 10), bool)
    mask[4:6, 4:6] = True
    filled = SimpleInpainter().inpaint(frame, mask)
    assert filled.dtype == np.uint8
    assert not np.isnan(filled.astype(float)).any()
    # filled region takes on neighbouring values, not left at zero-with-mask
    assert filled[4:6, 4:6].sum() > 0


def test_inpaint_noop_when_no_holes():
    frame = np.full((6, 6, 3), 100, np.uint8)
    out = SimpleInpainter().inpaint(frame, np.zeros((6, 6), bool))
    np.testing.assert_array_equal(out, frame)


# ------------------------------- beacons ---------------------------------


def test_beacon_carry_phase_attaches_to_gripper(episode_factory):
    ep = episode_factory(seed=4)
    scene = PointSplatReconstructor(puck_color=PUCK_COLOR).reconstruct(ep)
    tracker = FKBeaconTracker(gripper_xyz_channels=(0, 1, 2), grip_open_channel=3, carry_low=26, carry_high=36)
    # state channel 3 = 30 -> in carry band -> puck rides the gripper
    assert tracker.is_carrying(ep, 0)
    puck = tracker.puck_world(ep, 0, scene)
    grip = tracker.gripper_world(ep, 0)
    np.testing.assert_allclose(puck, grip, atol=1e-9)


def test_beacon_returns_none_without_state():
    ep = Episode(frames=np.zeros((2, 8, 8, 3), np.uint8))
    tracker = FKBeaconTracker(gripper_xyz_channels=(0, 1, 2))
    assert tracker.gripper_world(ep, 0) is None


# ------------------------------ generator --------------------------------


def test_generator_disabled_returns_empty(episode_factory):
    gen = MiniWorldGenerator(MiniWorldConfig(enabled=False))
    assert gen.generate([episode_factory()]) == []


def test_generator_produces_synthetic_with_provenance(episode_factory):
    reals = [episode_factory(seed=s) for s in range(2)]
    cfg = MiniWorldConfig(enabled=True, n_synthetic=3, puck_color=PUCK_COLOR,
                          object_pose_region=((-0.05, -0.05), (0.05, 0.05)), seed=11)
    out = MiniWorldGenerator(cfg).generate(reals)
    assert len(out) == 3
    for ep in out:
        assert ep.is_synthetic
        rec = ep.metadata["miniworld"]
        assert "camera_offset" in rec and "inpainted_fraction" in rec
        assert 0.0 <= rec["inpainted_fraction"] <= 1.0
        assert ep.frames.shape == reals[0].frames.shape
        assert ep.camera_poses is not None  # novel poses recorded


def test_generator_reproducible_with_seed(episode_factory):
    reals = [episode_factory(seed=5)]
    cfg = MiniWorldConfig(enabled=True, n_synthetic=2, puck_color=PUCK_COLOR,
                          object_pose_region=((-0.05, -0.05), (0.05, 0.05)), seed=99)
    a = MiniWorldGenerator(cfg).generate(reals)
    b = MiniWorldGenerator(cfg).generate(reals)
    for ea, eb in zip(a, b):
        np.testing.assert_array_equal(ea.frames, eb.frames)


def test_generator_skips_episodes_without_poses():
    plain = Episode(frames=np.zeros((3, 16, 16, 3), np.uint8))
    gen = MiniWorldGenerator(MiniWorldConfig(enabled=True, n_synthetic=2))
    assert gen.generate([plain]) == []


def test_generator_annotates_beacons_when_tracker_given(episode_factory):
    reals = [episode_factory(seed=6)]
    cfg = MiniWorldConfig(enabled=True, n_synthetic=1, puck_color=PUCK_COLOR, seed=3)
    tracker = FKBeaconTracker(gripper_xyz_channels=(0, 1, 2), grip_open_channel=3)
    out = MiniWorldGenerator(cfg, beacons=tracker).generate(reals)
    rec = out[0].metadata["miniworld"]
    assert "beacons" in rec
    assert len(rec["beacons"]["gripper_uv"]) == reals[0].num_frames


def test_generator_preserves_source_frame_representation(episode_factory):
    """Synthetic frames must match the source dtype AND value range, so float
    episodes in [0, 1] don't come back 255x too bright (regression test)."""
    base = episode_factory(seed=7)

    # float32 in [0, 1]: synthetic frames must stay in [0, 1], not [0, 255].
    float_source = Episode(
        frames=(base.frames.astype(np.float32) / 255.0),
        state=base.state, camera_poses=base.camera_poses, intrinsics=base.intrinsics, task=base.task,
    )
    cfg = MiniWorldConfig(enabled=True, n_synthetic=1, puck_color=PUCK_COLOR, seed=0)
    syn = MiniWorldGenerator(cfg).generate([float_source])[0]
    assert syn.frames.dtype == np.float32
    assert syn.frames.max() <= 1.0 + 1e-6
    assert syn.frames.min() >= 0.0

    # uint8 source stays uint8 in [0, 255].
    uint_syn = MiniWorldGenerator(cfg).generate([base])[0]
    assert uint_syn.frames.dtype == np.uint8
    assert uint_syn.frames.max() <= 255
