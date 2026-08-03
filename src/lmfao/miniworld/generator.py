from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from lmfao.datasets import Episode
from lmfao.miniworld.camera import Camera
from lmfao.miniworld.config import CameraOffset, MiniWorldConfig
from lmfao.miniworld.imaging import to_source_frame
from lmfao.miniworld.inpaint import SimpleInpainter
from lmfao.miniworld.interfaces import BeaconTracker, Inpainter, NovelViewRenderer, SceneReconstructor
from lmfao.miniworld.reconstruct import PointSplatReconstructor
from lmfao.miniworld.render import PointSplatRenderer
from lmfao.miniworld.scene import Scene


@dataclass
class MiniWorldGenerator:
    """Manufacture synthetic episodes from real ones.

    This is the GENERATE half of the system: it reconstructs a scene from a real
    episode (once, cached), then re-renders that episode's trajectory from novel
    camera viewpoints and/or with the puck moved, emitting new episodes in the
    same :class:`Episode` shape. Every synthetic episode is stamped with
    provenance under ``metadata["miniworld"]`` (plan §8d) and flagged
    ``metadata["synthetic"] = True`` so the pipeline stays source-blind.

    The four collaborators are the plan's interface seams; each defaults to the
    numpy reference implementation but can be swapped for a RoboSplat/torch
    backend without touching this orchestrator.
    """

    config: MiniWorldConfig
    reconstructor: SceneReconstructor | None = None
    renderer: NovelViewRenderer | None = None
    inpainter: Inpainter | None = None
    beacons: BeaconTracker | None = None

    def __post_init__(self) -> None:
        if self.reconstructor is None:
            self.reconstructor = PointSplatReconstructor(
                table_z=self.config.table_z,
                background_depth=self.config.background_depth,
                pixel_stride=self.config.pixel_stride,
                max_frames=self.config.max_frames,
                puck_color=self.config.puck_color,
                puck_color_tol=self.config.puck_color_tol,
            )
        if self.renderer is None:
            self.renderer = PointSplatRenderer(background=self.config.background)
        if self.inpainter is None:
            self.inpainter = SimpleInpainter()
        # beacons stay optional; None means "don't annotate".

    def generate(self, episodes: Iterable[Episode]) -> list[Episode]:
        """Produce up to ``config.n_synthetic`` synthetic episodes.

        Work is spread round-robin across the provided real episodes and the
        configured camera offsets, so a small demo set still yields varied views.
        """
        if not self.config.enabled or self.config.n_synthetic <= 0:
            return []

        sources = [ep for ep in episodes if self._is_renderable(ep)]
        if not sources:
            return []

        offsets = self.config.default_offsets()
        rng = np.random.default_rng(self.config.seed)
        scene_cache: dict[int, Scene] = {}

        synthetic: list[Episode] = []
        for i in range(self.config.n_synthetic):
            # Decouple the two indices so consecutive episodes walk the full
            # (source, offset) cartesian product before repeating; advancing both
            # in lockstep would emit byte-identical duplicates whenever
            # n_synthetic exceeds lcm(n_sources, n_offsets). The camera offset is
            # the dominant source of visual variation, so vary it on the fast
            # axis to maximise distinct views even when sources look alike.
            source = sources[(i // len(offsets)) % len(sources)]
            offset = offsets[i % len(offsets)]
            key = id(source)
            if key not in scene_cache:
                scene_cache[key] = self.reconstructor.reconstruct(source)
            scene = scene_cache[key]
            puck_shift = self._sample_puck_shift(rng)
            synthetic.append(self._render_episode(source, scene, offset, puck_shift, variant=i))
        return synthetic

    def reconstruct(self, episode: Episode) -> Scene:
        """Expose reconstruction for callers that want the scene directly."""
        return self.reconstructor.reconstruct(episode)

    def _is_renderable(self, episode: Episode) -> bool:
        return episode.camera_poses is not None and episode.intrinsics is not None

    def _sample_puck_shift(self, rng: np.random.Generator) -> np.ndarray | None:
        region = self.config.object_pose_region
        if region is None:
            return None
        (lo_x, lo_y), (hi_x, hi_y) = region
        return np.array([rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y), 0.0])

    def _render_episode(
        self,
        source: Episode,
        scene: Scene,
        offset: CameraOffset,
        puck_shift: np.ndarray | None,
        variant: int,
    ) -> Episode:
        moved_puck = None
        if puck_shift is not None and not scene.puck.is_empty:
            moved_puck = scene.puck.translated(puck_shift)
        # The shift only actually happened if a puck was segmented to move; record
        # None otherwise so provenance never claims an edit that had no effect.
        applied_shift = puck_shift if moved_puck is not None else None
        cloud = scene.render_cloud(moved_puck)

        num_frames = source.num_frames
        out_frames = np.empty_like(source.frames)
        novel_poses = np.empty((num_frames, 4, 4))
        hole_fracs = np.empty(num_frames)
        gripper_uv: list[list[float] | None] = []
        puck_uv: list[list[float] | None] = []

        for f in range(num_frames):
            real_cam = Camera(source.intrinsics, source.camera_poses[f], source.width, source.height)
            novel_cam = real_cam.offset(offset.translation, offset.yaw, offset.pitch)
            novel_poses[f] = novel_cam.pose

            rgb, hole = self.renderer.render(cloud, novel_cam)
            hole_fracs[f] = float(hole.mean())
            filled = self.inpainter.inpaint(rgb, hole)
            out_frames[f] = to_source_frame(filled, source.frames)

            gripper_uv.append(self._beacon_uv(self._gripper_world(source, f), novel_cam))
            puck_uv.append(self._beacon_uv(self._puck_world(source, f, scene, moved_puck, puck_shift), novel_cam))

        metadata = self._provenance(source, offset, applied_shift, hole_fracs, gripper_uv, puck_uv, variant)
        return Episode(
            frames=out_frames,
            state=None if source.state is None else source.state.copy(),
            actions=None if source.actions is None else source.actions.copy(),
            fps=source.fps,
            task=source.task,
            camera_poses=novel_poses,
            intrinsics=source.intrinsics.copy(),
            metadata=metadata,
        )

    def _gripper_world(self, source: Episode, frame: int) -> np.ndarray | None:
        if self.beacons is None:
            return None
        return self.beacons.gripper_world(source, frame)

    def _puck_world(
        self,
        source: Episode,
        frame: int,
        scene: Scene,
        moved_puck,
        puck_shift: np.ndarray | None,
    ) -> np.ndarray | None:
        if self.beacons is None:
            return None
        world = self.beacons.puck_world(source, frame, scene)
        if world is None:
            return None
        # If the puck's start pose was edited and it is not being carried, the
        # beacon moves with it.
        if puck_shift is not None and moved_puck is not None:
            carrying = getattr(self.beacons, "is_carrying", lambda *_: False)(source, frame)
            if not carrying:
                world = world + puck_shift
        return world

    @staticmethod
    def _beacon_uv(world: np.ndarray | None, camera: Camera) -> list[float] | None:
        if world is None:
            return None
        uv, z = camera.project(world[None, :])
        if z[0] <= 0:
            return None
        return [float(uv[0, 0]), float(uv[0, 1])]

    def _provenance(
        self,
        source: Episode,
        offset: CameraOffset,
        puck_shift: np.ndarray | None,
        hole_fracs: np.ndarray,
        gripper_uv: list,
        puck_uv: list,
        variant: int,
    ) -> dict:
        # Deep-copy carried-over values so a synthetic episode never aliases the
        # source's nested mutables (state/actions/intrinsics are already copied).
        metadata = {
            k: copy.deepcopy(v)
            for k, v in source.metadata.items()
            if k not in ("miniworld", "synthetic")
        }
        record = {
            "source_task": source.task,
            "variant": int(variant),
            "camera_offset": {
                "translation": list(offset.translation),
                "yaw": offset.yaw,
                "pitch": offset.pitch,
            },
            "object_pose_edit": None if puck_shift is None else list(map(float, puck_shift)),
            "inpainted_fraction": float(hole_fracs.mean()),
            "max_inpainted_fraction": float(hole_fracs.max()) if hole_fracs.size else 0.0,
        }
        if self.beacons is not None:
            record["beacons"] = {"gripper_uv": gripper_uv, "puck_uv": puck_uv}
        metadata["synthetic"] = True
        metadata["miniworld"] = record
        return metadata
