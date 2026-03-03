"""Stable-Baselines3 callbacks for Sisyphus training.

CurriculumCallback — manages slope/mass transitions.
TrajectoryRenderCallback — periodic trajectory logging + preview rendering.
"""

import os
import logging

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from train.curriculum import CurriculumManager
from logging_utils.trajectory_logger import TrajectoryLogger

logger = logging.getLogger(__name__)


class CurriculumCallback(BaseCallback):
    """Update environment curriculum parameters during training."""

    def __init__(self, curriculum: CurriculumManager, verbose: int = 1):
        super().__init__(verbose)
        self.curriculum = curriculum
        self._last_phase = None

    def _on_step(self) -> bool:
        total_steps = self.num_timesteps
        changed, params = self.curriculum.check_transition(total_steps)

        if changed or self._last_phase is None:
            if self.verbose:
                logger.info(
                    f"Curriculum transition at step {total_steps}: "
                    f"Phase {params.phase} (slope={params.slope_deg}°, "
                    f"mass={params.rock_mass}kg, infinite={params.infinite})"
                )

            # Update all envs in the vectorized environment
            self.training_env.env_method(
                "set_curriculum_params",
                params.slope_deg,
                params.rock_mass,
                params.infinite,
                params.alive_bonus,
                params.upright_coef,
                params.forward_push_coef,
            )

            # Log to tensorboard
            self.logger.record("curriculum/phase", params.phase)
            self.logger.record("curriculum/slope_deg", params.slope_deg)
            self.logger.record("curriculum/rock_mass", params.rock_mass)
            self.logger.record("curriculum/infinite", int(params.infinite))
            self.logger.record("curriculum/alive_bonus", params.alive_bonus)
            self.logger.record("curriculum/upright_coef", params.upright_coef)
            self.logger.record(
                "curriculum/forward_push_coef", params.forward_push_coef
            )

            self._last_phase = params.phase

        return True


class TrajectoryRenderCallback(BaseCallback):
    """Periodically evaluate, log trajectory, and render preview MP4."""

    def __init__(
        self,
        eval_env,
        save_freq: int = 500_000,
        replay_dir: str = "replays",
        render_dir: str = "renders_preview",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.save_freq = save_freq
        self.replay_dir = replay_dir
        self.render_dir = render_dir
        self._checkpoint_counter = 0
        self._episode_counter = 0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq != 0:
            return True

        if self.verbose:
            logger.info(f"Recording trajectory at step {self.num_timesteps}")

        checkpoint_id = self._checkpoint_counter
        self._checkpoint_counter += 1

        # Run one evaluation episode with logging
        traj_logger = TrajectoryLogger(save_dir=self.replay_dir)
        obs, info = self.eval_env.reset()

        rock_body_id = self.eval_env._rock_id
        model = self.eval_env.model

        for step in range(self.eval_env.max_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)

            traj_logger.record_step(
                model=model,
                data=self.eval_env.data,
                action=action,
                reward=reward,
                rock_body_id=rock_body_id,
                height_offset=info.get("total_height_accumulated", 0.0),
            )

            if terminated or truncated:
                break

        # Save trajectory
        metadata = {
            "slope": self.eval_env._slope_deg,
            "mass": float(model.body_mass[self.eval_env._rock_id]),
            "checkpoint": checkpoint_id,
            "total_steps": self.num_timesteps,
            "timestep": float(model.opt.timestep),
            "fps": 1.0 / (model.opt.timestep * 5),  # env step rate
        }
        traj_path = traj_logger.save_episode(
            episode_id=self._episode_counter,
            checkpoint_id=checkpoint_id,
            metadata=metadata,
        )
        self._episode_counter += 1

        if self.verbose:
            logger.info(f"Saved trajectory: {traj_path}")

        # Render preview MP4
        try:
            from render.preview_renderer import PreviewRenderer

            renderer = PreviewRenderer(model, width=1920, height=1080)
            mp4_path = os.path.join(
                self.render_dir,
                f"preview_step_{self.num_timesteps}.mp4",
            )
            os.makedirs(self.render_dir, exist_ok=True)
            renderer.render_trajectory(traj_path, mp4_path, fps=30)
            renderer.close()
            if self.verbose:
                logger.info(f"Saved preview: {mp4_path}")
        except Exception as e:
            logger.warning(f"Preview render failed: {e}")

        return True
