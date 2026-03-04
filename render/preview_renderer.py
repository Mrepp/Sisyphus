"""Fast preview renderer — MuJoCo offscreen rendering to MP4.

Renders trajectories from .npz files or live episodes for training progress
monitoring. Fixed wide-angle side camera, neutral lighting, 1080p.
"""

import os
import numpy as np
import mujoco
import imageio


class PreviewRenderer:
    def __init__(self, model: mujoco.MjModel, width: int = 1920, height: int = 1080):
        self.model = model
        self.data = mujoco.MjData(model)
        self.width = width
        self.height = height
        self._renderer = mujoco.Renderer(model, height=height, width=width)

        # Camera defaults
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = 7.0
        self.camera.azimuth = 90.0
        self.camera.elevation = -15.0
        self.camera.lookat[:] = [0, 0, 1.0]

    def _render_frame(self, lookat: np.ndarray | None = None) -> np.ndarray:
        """Render a single frame and return pixel array."""
        if lookat is not None:
            self.camera.lookat[:] = lookat
        self._renderer.update_scene(self.data, self.camera)
        return self._renderer.render().copy()

    def render_trajectory(
        self,
        trajectory_path: str,
        output_path: str,
        fps: int = 30,
        skip_frames: int = 1,
    ):
        """Render a saved .npz trajectory to MP4.

        Args:
            trajectory_path: Path to .npz trajectory file.
            output_path: Output MP4 path.
            fps: Output video framerate.
            skip_frames: Render every Nth frame (for speed).
        """
        from logging_utils.trajectory_logger import TrajectoryLogger

        traj = TrajectoryLogger.load(trajectory_path)
        qpos_seq = traj["qpos"]
        T = len(qpos_seq)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        writer = imageio.get_writer(
            output_path, fps=fps, codec="libx264",
            format="FFMPEG", macro_block_size=1,
        )

        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")

        try:
            for t in range(0, T, skip_frames):
                self.data.qpos[:] = qpos_seq[t]
                mujoco.mj_forward(self.model, self.data)

                lookat = self.data.xpos[torso_id].copy()
                frame = self._render_frame(lookat)
                writer.append_data(frame)
        finally:
            writer.close()

        return output_path

    def render_episode(
        self,
        env,
        policy,
        output_path: str,
        fps: int = 30,
        max_steps: int = 1000,
        deterministic: bool = True,
        trajectory_path: str | None = None,
        trajectory_metadata: dict | None = None,
    ) -> tuple[str, str | None]:
        """Run one evaluation episode, render each frame, save as MP4.

        Optionally records full simulation state as a .npz trajectory
        suitable for Blender export.

        Args:
            env: A SisyphusEnv instance (unwrapped).
            policy: SB3 model with .predict() method.
            output_path: Output MP4 path.
            fps: Output framerate.
            max_steps: Max steps per episode.
            deterministic: Use deterministic policy.
            trajectory_path: If set, save trajectory .npz to this path.
            trajectory_metadata: Optional metadata dict stored in the .npz.

        Returns:
            (mp4_path, trajectory_path or None)
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        writer = imageio.get_writer(
            output_path, fps=fps, codec="libx264",
            format="FFMPEG", macro_block_size=1,
        )

        # Set up trajectory logger if requested
        logger = None
        if trajectory_path is not None:
            from logging_utils.trajectory_logger import TrajectoryLogger
            traj_dir = os.path.dirname(trajectory_path) or "."
            logger = TrajectoryLogger(save_dir=traj_dir)

        obs, _ = env.reset()
        torso_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "torso")

        # Use the env's model/data for rendering
        self.data = env.data
        self.model = env.model
        self._renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)

        death_frames = int(3.0 * fps)  # 3 seconds of post-death footage
        zero_action = np.zeros(env.action_space.shape)

        try:
            for step in range(max_steps):
                lookat = env.data.xpos[torso_id].copy()
                frame = self._render_frame(lookat)
                writer.append_data(frame)

                action, _ = policy.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)

                if logger is not None:
                    logger.record_step(
                        model=env.model,
                        data=env.data,
                        action=action,
                        reward=reward,
                        rock_body_id=env._rock_id,
                        height_offset=env._total_height_accumulated,
                    )

                if terminated or truncated:
                    # Continue rendering with zero actions so the
                    # ragdoll / rock settle visibly for 3 more seconds.
                    for _ in range(death_frames):
                        env.data.ctrl[:] = 0.0
                        mujoco.mj_step(env.model, env.data, nstep=5)

                        lookat = env.data.xpos[torso_id].copy()
                        frame = self._render_frame(lookat)
                        writer.append_data(frame)

                        if logger is not None:
                            logger.record_step(
                                model=env.model,
                                data=env.data,
                                action=zero_action,
                                reward=0.0,
                                rock_body_id=env._rock_id,
                                height_offset=env._total_height_accumulated,
                            )
                    break
        finally:
            writer.close()

        # Save trajectory
        saved_traj = None
        if logger is not None and len(logger._qpos) > 0:
            meta = trajectory_metadata or {}
            meta["fps"] = fps
            # Save directly to the requested path
            traj_filename = os.path.basename(trajectory_path)
            logger.save_dir = os.path.dirname(trajectory_path) or "."
            # Use save_episode with dummy IDs, then rename
            temp_path = logger.save_episode(
                episode_id=0, checkpoint_id=0, metadata=meta,
            )
            target = trajectory_path if trajectory_path.endswith(".npz") else trajectory_path + ".npz"
            if temp_path != target:
                os.replace(temp_path, target)
            saved_traj = target

        return output_path, saved_traj

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
