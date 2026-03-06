"""Stable-Baselines3 callbacks for Sisyphus training.

CurriculumCallback — manages metric-gated phase promotion.
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
    """Update environment curriculum parameters during training.

    Collects per-episode promotion_score from info dicts and feeds
    them to CurriculumManager for metric-gated phase promotion.
    """

    def __init__(
        self,
        curriculum: CurriculumManager,
        eval_env=None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.curriculum = curriculum
        self.eval_env = eval_env
        self._last_phase = None
        self._last_walk_only = None

    def _apply_params(self, params):
        """Push curriculum parameters to all vectorized envs."""
        self.training_env.env_method(
            "set_curriculum_params",
            params.slope_deg,
            params.rock_mass,
            params.infinite,
            params.alive_bonus,
            params.upright_coef,
            params.forward_push_coef,
            params.walk_only_mode,
        )
        # Also sync eval env if provided
        if self.eval_env is not None:
            self.eval_env.set_curriculum_params(
                params.slope_deg,
                params.rock_mass,
                params.infinite,
                params.alive_bonus,
                params.upright_coef,
                params.forward_push_coef,
                params.walk_only_mode,
            )

    def _log_curriculum(self, params):
        """Log curriculum state to TensorBoard."""
        self.logger.record("curriculum/phase", params.phase)
        self.logger.record("curriculum/slope_deg", params.slope_deg)
        self.logger.record("curriculum/rock_mass", params.rock_mass)
        self.logger.record("curriculum/infinite", int(params.infinite))
        self.logger.record("curriculum/alive_bonus", params.alive_bonus)
        self.logger.record("curriculum/upright_coef", params.upright_coef)
        self.logger.record(
            "curriculum/forward_push_coef", params.forward_push_coef
        )
        self.logger.record(
            "curriculum/walk_only_mode", int(params.walk_only_mode)
        )

    def _on_step(self) -> bool:
        total_steps = self.num_timesteps

        # --- Collect episode completion metrics ---
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for i, info in enumerate(infos):
            # SB3 >=2.1 wraps terminal info in "terminal_info" key;
            # SB3 >=2.3 provides it directly when done=True.
            # The fallback `info.get("terminal_info", info)` handles both.
            if dones is not None and i < len(dones) and dones[i]:
                # Check for terminal_observation wrapper pattern
                terminal_info = info.get("terminal_info", info)
                score = terminal_info.get("promotion_score")
                if score is not None:
                    self.curriculum.add_promotion_score(score)
                contact_frac = terminal_info.get("contact_fraction")
                if contact_frac is not None:
                    self.logger.record("metrics/contact_fraction", contact_frac)

        # --- Check for phase transition ---
        changed, params = self.curriculum.check_transition(total_steps)

        if changed:
            if self.verbose:
                logger.info(
                    f"Curriculum PROMOTION at step {total_steps}: "
                    f"Phase {params.phase} "
                    f"(slope={params.slope_deg}deg, "
                    f"mass={params.rock_mass}kg, "
                    f"infinite={params.infinite})"
                )
                rolling = self.curriculum.get_rolling_promotion_score()
                if rolling is not None:
                    logger.info(
                        f"  Rolling promotion score: {rolling:.2f}"
                    )
            self._apply_params(params)
            self._log_curriculum(params)
            self._last_phase = params.phase
            self._last_walk_only = params.walk_only_mode

        elif self._last_phase is None:
            # First step — apply initial params
            params = self.curriculum.get_params(total_steps)
            self._apply_params(params)
            self._log_curriculum(params)
            self._last_phase = params.phase
            self._last_walk_only = params.walk_only_mode

        else:
            # Check for walk-only mode transition within a phase
            params = self.curriculum.get_params(total_steps)
            if params.walk_only_mode != self._last_walk_only:
                if self.verbose:
                    mode = "WALK-ONLY" if params.walk_only_mode else "FULL"
                    logger.info(
                        f"Phase {params.phase} mode change at "
                        f"step {total_steps}: {mode}"
                    )
                self._apply_params(params)
                self._log_curriculum(params)
                self._last_walk_only = params.walk_only_mode

        # --- Log detailed metrics every 10000 steps ---
        if self.num_timesteps % 10000 == 0:
            if infos:
                self.logger.record(
                    "metrics/rock_distance_mean",
                    np.mean([i.get("rock_distance", 0)
                             for i in infos]))
                self.logger.record(
                    "metrics/torso_up_dot_mean",
                    np.mean([i.get("torso_up_dot", 0)
                             for i in infos]))
                self.logger.record(
                    "metrics/com_vel_x_mean",
                    np.mean([i.get("com_vel_x", 0)
                             for i in infos]))
                self.logger.record(
                    "metrics/rock_delta_x_mean",
                    np.mean([i.get("rock_delta_x", 0)
                             for i in infos]))
                self.logger.record(
                    "metrics/hand_contact_rate",
                    np.mean([i.get("hand_contact", 0)
                             for i in infos]))
                # Locomotion metrics
                self.logger.record(
                    "metrics/cadence_reward_mean",
                    np.mean([i.get("cadence_reward", 0)
                             for i in infos]))
                self.logger.record(
                    "metrics/stance_reward_mean",
                    np.mean([i.get("stance_reward", 0)
                             for i in infos]))
                self.logger.record(
                    "metrics/com_stability_mean",
                    np.mean([i.get("com_stability_reward", 0)
                             for i in infos]))
                self.logger.record(
                    "metrics/lean_reward_mean",
                    np.mean([i.get("lean_reward", 0)
                             for i in infos]))
                # Penalty metrics
                self.logger.record(
                    "metrics/cot_penalty_mean",
                    np.mean([i.get("cot_penalty", 0)
                             for i in infos]))
                self.logger.record(
                    "metrics/smoothness_penalty_mean",
                    np.mean([i.get("smoothness_penalty", 0)
                             for i in infos]))
                self.logger.record(
                    "metrics/lateral_penalty_mean",
                    np.mean([i.get("lateral_penalty", 0)
                             for i in infos]))
                self.logger.record(
                    "metrics/rock_rollback_penalty_mean",
                    np.mean([i.get("rock_rollback_penalty", 0)
                             for i in infos]))
                # Push metrics
                self.logger.record(
                    "metrics/force_gate_mean",
                    np.mean([i.get("force_gate", 0)
                             for i in infos]))
                self.logger.record(
                    "metrics/rock_contact_force_mean",
                    np.mean([i.get("rock_contact_force", 0)
                             for i in infos]))
                self.logger.record(
                    "metrics/gait_step_count_mean",
                    np.mean([i.get("gait_step_count", 0)
                             for i in infos]))
                # Contact metrics
                self.logger.record(
                    "metrics/right_foot_on_rate",
                    np.mean([i.get("right_foot_on", 0)
                             for i in infos]))
                self.logger.record(
                    "metrics/left_foot_on_rate",
                    np.mean([i.get("left_foot_on", 0)
                             for i in infos]))
                self.logger.record(
                    "metrics/rock_y_mean",
                    np.mean([i.get("rock_y", 0)
                             for i in infos]))
                self.logger.record(
                    "metrics/agent_touching_rock_rate",
                    np.mean([i.get("agent_touching_rock", 0)
                             for i in infos]))
                self.logger.record(
                    "metrics/torso_height_mean",
                    np.mean([i.get("torso_height", 0)
                             for i in infos]))
                # Posture reward metrics
                self.logger.record(
                    "metrics/height_reward_posture_mean",
                    np.mean([i.get("height_reward_posture", 0)
                             for i in infos]))
                self.logger.record(
                    "metrics/getup_reward_mean",
                    np.mean([i.get("getup_reward", 0)
                             for i in infos]))
                self.logger.record(
                    "metrics/cadence_continuous_mean",
                    np.mean([i.get("cadence_continuous", 0)
                             for i in infos]))
                self.logger.record(
                    "metrics/symmetry_penalty_mean",
                    np.mean([i.get("symmetry_penalty", 0)
                             for i in infos]))
                self.logger.record(
                    "metrics/balance_reward_mean",
                    np.mean([i.get("balance_reward", 0)
                             for i in infos]))

                # Rolling promotion score
                rolling = self.curriculum.get_rolling_promotion_score()
                if rolling is not None:
                    self.logger.record(
                        "curriculum/rolling_promotion_score",
                        rolling)

        return True


class TrajectoryRenderCallback(BaseCallback):
    """Periodically evaluate, log trajectory, and render preview MP4."""

    def __init__(
        self,
        eval_env,
        eval_vec_env=None,
        training_env=None,
        save_freq: int = 500_000,
        replay_dir: str = "replays",
        render_dir: str = "renders_preview",
        render_enabled: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_vec_env = eval_vec_env
        self._train_vec_env = training_env
        self.save_freq = save_freq
        self.replay_dir = replay_dir
        self.render_dir = render_dir
        self.render_enabled = render_enabled
        self._checkpoint_counter = 0
        self._episode_counter = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq != 0:
            return True

        if self.verbose:
            logger.info(
                f"Recording trajectory at step "
                f"{self.num_timesteps}"
            )

        checkpoint_id = self._checkpoint_counter
        self._checkpoint_counter += 1

        # Sync normalization stats from training env
        if (self.eval_vec_env is not None
                and self._train_vec_env is not None):
            from stable_baselines3.common.vec_env import (
                sync_envs_normalization,
            )
            sync_envs_normalization(
                self._train_vec_env, self.eval_vec_env,
            )

        # Choose inference path: normalized or raw
        use_vec = self.eval_vec_env is not None

        traj_logger = TrajectoryLogger(
            save_dir=self.replay_dir,
        )

        if use_vec:
            obs = self.eval_vec_env.reset()
        else:
            obs, info = self.eval_env.reset()

        rock_body_id = self.eval_env._rock_id
        mj_model = self.eval_env.model

        for step in range(self.eval_env.max_steps):
            action, _ = self.model.predict(
                obs, deterministic=True,
            )

            if use_vec:
                obs, reward, done, infos = (
                    self.eval_vec_env.step(action)
                )
                finished = done[0]
                info = infos[0]
                act = action[0]
                rew = reward[0]
            else:
                obs, rew, terminated, truncated, info = (
                    self.eval_env.step(action)
                )
                finished = terminated or truncated
                act = action

            traj_logger.record_step(
                model=mj_model,
                data=self.eval_env.data,
                action=act,
                reward=rew,
                rock_body_id=rock_body_id,
                height_offset=info.get(
                    "total_height_accumulated", 0.0
                ),
            )

            if finished:
                break

        # Save trajectory
        metadata = {
            "slope": self.eval_env._slope_deg,
            "mass": float(
                mj_model.body_mass[
                    self.eval_env._rock_id
                ]
            ),
            "checkpoint": checkpoint_id,
            "total_steps": self.num_timesteps,
            "timestep": float(mj_model.opt.timestep),
            "fps": 1.0 / (mj_model.opt.timestep * 5),
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
        if self.render_enabled:
            try:
                from render.preview_renderer import (
                    PreviewRenderer,
                )

                renderer = PreviewRenderer(
                    mj_model, width=1920, height=1080,
                )
                mp4_path = os.path.join(
                    self.render_dir,
                    f"preview_step_"
                    f"{self.num_timesteps}.mp4",
                )
                os.makedirs(
                    self.render_dir, exist_ok=True,
                )
                renderer.render_trajectory(
                    traj_path, mp4_path, fps=30,
                )
                renderer.close()
                if self.verbose:
                    logger.info(
                        f"Saved preview: {mp4_path}"
                    )
            except Exception as e:
                logger.warning(
                    f"Preview render failed: {e}"
                )

        return True
