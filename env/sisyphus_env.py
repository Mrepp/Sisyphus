"""Sisyphus Humanoid Environment — MuJoCo + Gymnasium.

A humanoid pushes a rock up a heightfield incline. Supports curriculum
learning (variable slope/mass) and an infinite-illusion mode for Phase IV.
"""

import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "sisyphus_humanoid.xml")


class SisyphusEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 67}
    _ROCK_RADIUS = 0.7  # metres — boulder radius (top reaches ~1.4m ≈ shoulder height)

    def __init__(
        self,
        slope_deg: float = 0.0,
        rock_mass: float = 40.0,
        max_steps: int = 1000,
        infinite_mode: bool = False,
        render_mode: str | None = None,
        alive_bonus: float = 2.0,
        upright_coef: float = 1.0,
        forward_push_coef: float = 5.0,
        hand_proximity_coef: float = 0.5,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.infinite_mode = infinite_mode
        self._slope_deg = slope_deg
        self._alive_bonus = alive_bonus
        self._upright_coef = upright_coef
        self._forward_push_coef = forward_push_coef
        self._hand_proximity_coef = hand_proximity_coef

        # Load model
        model_path = os.path.normpath(_MODEL_PATH)
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Cache body / joint IDs
        self._torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self._rock_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rock")
        self._rock_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "rock_geom")
        self._terrain_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_geom")
        self._right_hand_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_hand")
        self._left_hand_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_hand")
        self._HAND_RADIUS = 0.04
        self._hfield_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")

        # Terrain geometry
        self._hfield_nrow = self.model.hfield_nrow[self._hfield_id]
        self._hfield_ncol = self.model.hfield_ncol[self._hfield_id]
        self._hfield_size = self.model.hfield_size[self._hfield_id].copy()  # (x_half, y_half, z_max, z_base)
        self._terrain_length = self._hfield_size[0] * 2  # full x extent

        # Rock joint: the free joint adds 7 qpos (pos + quat) and 6 qvel
        # Find the rock's qpos/qvel address
        self._rock_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rock_joint")
        self._rock_qpos_adr = self.model.jnt_qposadr[self._rock_jnt_id]
        self._rock_qvel_adr = self.model.jnt_dofadr[self._rock_jnt_id]

        # Humanoid root joint
        self._root_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        self._root_qpos_adr = self.model.jnt_qposadr[self._root_jnt_id]
        self._root_qvel_adr = self.model.jnt_dofadr[self._root_jnt_id]

        # Number of actuators
        self._nu = self.model.nu  # 17

        # Set rock mass
        self._set_rock_mass(rock_mass)

        # Set terrain slope
        self._set_slope(slope_deg)

        # Forward to compute derived quantities
        mujoco.mj_forward(self.model, self.data)

        # --- Spaces ---
        # Observation: humanoid qpos (skip root x), humanoid qvel, rock rel pos/vel, torso height, com vel
        obs_size = self._get_obs().shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self._nu,), dtype=np.float64
        )

        # State tracking
        self._step_count = 0
        self._prev_rock_height = 0.0
        self._prev_rock_x = 0.0
        self._total_height_accumulated = 0.0
        self._reset_x = 1.0  # x position to teleport back to

        # Renderer (lazy init)
        self._renderer = None

    # ------------------------------------------------------------------
    # Curriculum interface
    # ------------------------------------------------------------------
    def set_curriculum_params(
        self,
        slope_deg: float,
        rock_mass: float,
        infinite_mode: bool = False,
        alive_bonus: float = 0.0,
        upright_coef: float = 0.0,
        forward_push_coef: float = 5.0,
    ):
        self._slope_deg = slope_deg
        self.infinite_mode = infinite_mode
        self._alive_bonus = alive_bonus
        self._upright_coef = upright_coef
        self._forward_push_coef = forward_push_coef
        self._set_rock_mass(rock_mass)
        self._set_slope(slope_deg)

    # ------------------------------------------------------------------
    # Terrain
    # ------------------------------------------------------------------
    def _set_slope(self, slope_deg: float):
        """Populate heightfield with a linear incline."""
        nrow = self._hfield_nrow
        ncol = self._hfield_ncol
        z_max = self._hfield_size[2]

        slope_rad = np.radians(slope_deg)
        # Row 0 = far end (+x), row nrow-1 = near end (0).  MuJoCo hfield:
        # row index 0 corresponds to +x, row nrow-1 to -x.
        # We want height increasing with +x (uphill direction).
        row_heights = np.linspace(0, 1, nrow)  # 0 at near, 1 at far

        # Scale so that max physical height = tan(slope) * terrain_length
        max_h = np.tan(slope_rad) * self._terrain_length
        # Clamp to hfield z_max (normalised values are 0-1 in hfield_data)
        if z_max > 0:
            row_normalized = row_heights * min(max_h / z_max, 1.0)
        else:
            row_normalized = np.zeros(nrow)

        # Tile across columns
        hfield_data = np.tile(row_normalized[:, None], (1, ncol)).flatten().astype(np.float32)

        # Write into model
        start = self.model.hfield_adr[self._hfield_id]
        self.model.hfield_data[start: start + nrow * ncol] = hfield_data

    def _terrain_height_at_x(self, x: float) -> float:
        """Approximate terrain z at a given x coordinate."""
        slope_rad = np.radians(self._slope_deg)
        return np.tan(slope_rad) * max(x, 0.0)

    # ------------------------------------------------------------------
    # Rock mass
    # ------------------------------------------------------------------
    def _set_rock_mass(self, mass: float):
        self.model.body_mass[self._rock_id] = mass

    # ------------------------------------------------------------------
    # Hand-rock helpers
    # ------------------------------------------------------------------
    def _hand_rock_distances(self) -> tuple[float, float]:
        """Surface-to-surface distance from each hand to the rock."""
        rock_pos = self.data.geom_xpos[self._rock_geom_id]
        r_pos = self.data.geom_xpos[self._right_hand_geom_id]
        l_pos = self.data.geom_xpos[self._left_hand_geom_id]
        r = self._HAND_RADIUS + self._ROCK_RADIUS
        right_d = max(np.linalg.norm(r_pos - rock_pos) - r, 0.0)
        left_d = max(np.linalg.norm(l_pos - rock_pos) - r, 0.0)
        return right_d, left_d

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        # Humanoid root position
        root_x = qpos[self._root_qpos_adr]
        root_y = qpos[self._root_qpos_adr + 1]
        root_z = qpos[self._root_qpos_adr + 2]

        # Skip root x for translation invariance; keep y, z, and quaternion + joints
        humanoid_qpos = np.concatenate([
            qpos[self._root_qpos_adr + 1: self._rock_qpos_adr]  # skip root_x
        ])
        humanoid_qvel = qvel[self._root_qvel_adr: self._rock_qvel_adr]

        # Rock position relative to torso
        torso_pos = self.data.xpos[self._torso_id]
        rock_pos = self.data.xpos[self._rock_id]
        rock_rel = rock_pos - torso_pos

        # Rock velocity
        rock_vel = qvel[self._rock_qvel_adr: self._rock_qvel_adr + 3]

        # Torso height
        torso_height = np.array([torso_pos[2]])

        # Center of mass velocity
        com_vel = self.data.subtree_linvel[0].copy()  # whole model COM velocity

        # Hand-to-rock surface distances
        r_hand_d, l_hand_d = self._hand_rock_distances()
        hand_dists = np.array([r_hand_d, l_hand_d])

        obs = np.concatenate([
            humanoid_qpos,
            humanoid_qvel,
            rock_rel,
            rock_vel,
            torso_height,
            com_vel,
            hand_dists,
        ])
        return obs

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        # Clip and apply action
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = action

        # Physics substeps
        mujoco.mj_step(self.model, self.data, nstep=5)

        self._step_count += 1

        # Current state
        torso_pos = self.data.xpos[self._torso_id].copy()
        rock_pos = self.data.xpos[self._rock_id].copy()
        torso_z = torso_pos[2]

        # Rock height (including accumulated offset for infinite mode)
        current_rock_height = rock_pos[2] + self._total_height_accumulated
        delta_h_rock = current_rock_height - self._prev_rock_height
        self._prev_rock_height = current_rock_height

        # Reward
        height_reward = 100.0 * delta_h_rock
        torque_penalty = 0.0001 * np.sum(action ** 2)
        fell = torso_z < 0.7
        fall_penalty = 500.0 if fell else 0.0

        # Forward-push reward: encourage +x rock movement, penalise -x
        rock_x = rock_pos[0]
        delta_x_rock = rock_x - self._prev_rock_x
        self._prev_rock_x = rock_x
        # Decay within episode so agent doesn't over-rely on shaping
        episode_progress = self._step_count / self.max_steps  # 0→1
        forward_decay = max(1.0 - episode_progress, 0.0)
        forward_reward = self._forward_push_coef * delta_x_rock * forward_decay

        # Posture scaffolding (decayed via curriculum — zero by Phase III)
        alive_bonus = self._alive_bonus
        torso_xmat = self.data.xmat[self._torso_id].reshape(3, 3)
        torso_up_dot = torso_xmat[2, 2]  # z-component of torso z-axis vs world up
        upright_bonus = self._upright_coef * torso_up_dot

        # Hand-proximity reward: encourage hands near/on the rock
        right_dist, left_dist = self._hand_rock_distances()
        hand_proximity = (
            np.exp(-3.0 * right_dist)
            + np.exp(-3.0 * left_dist)
        )
        hand_reward = self._hand_proximity_coef * hand_proximity

        reward = (height_reward + forward_reward
                  - torque_penalty - fall_penalty
                  + alive_bonus + upright_bonus
                  + hand_reward)

        # Infinite illusion: teleport when approaching terrain end
        if self.infinite_mode and torso_pos[0] > 0.7 * self._terrain_length:
            offset_x = torso_pos[0] - self._reset_x
            offset_z = self._terrain_height_at_x(torso_pos[0]) - self._terrain_height_at_x(self._reset_x)

            # Shift humanoid
            self.data.qpos[self._root_qpos_adr] -= offset_x
            self.data.qpos[self._root_qpos_adr + 2] -= offset_z

            # Shift rock
            self.data.qpos[self._rock_qpos_adr] -= offset_x
            self.data.qpos[self._rock_qpos_adr + 2] -= offset_z

            # Accumulate true height
            self._total_height_accumulated += offset_z

            # Update prev rock height to avoid false reward spike
            self._prev_rock_height = self.data.xpos[self._rock_id][2] + self._total_height_accumulated

            mujoco.mj_forward(self.model, self.data)

        # Termination
        terminated = bool(fell)
        truncated = self._step_count >= self.max_steps

        info = {
            "rock_height": current_rock_height,
            "total_height_accumulated": self._total_height_accumulated,
            "torso_height": torso_z,
            "torso_up_dot": torso_up_dot,
            "alive_bonus": alive_bonus,
            "upright_bonus": upright_bonus,
            "forward_reward": forward_reward,
            "hand_reward": hand_reward,
            "right_hand_dist": right_dist,
            "left_hand_dist": left_dist,
            "step_count": self._step_count,
        }

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Humanoid standing pose (default qpos from model)
        self.data.qpos[:] = self.model.key_qpos[0] if self.model.nkey > 0 else 0.0

        # If no keyframe, set basic standing position
        if self.model.nkey == 0:
            self.data.qpos[self._root_qpos_adr + 2] = 1.4  # z height

        # Randomise rock position slightly
        rock_x = self.np_random.uniform(1.3, 1.7)
        rock_z = self._terrain_height_at_x(rock_x) + self._ROCK_RADIUS  # radius above terrain
        self.data.qpos[self._rock_qpos_adr] = rock_x
        self.data.qpos[self._rock_qpos_adr + 1] = 0.0
        self.data.qpos[self._rock_qpos_adr + 2] = rock_z
        # Quaternion identity
        self.data.qpos[self._rock_qpos_adr + 3] = 1.0
        self.data.qpos[self._rock_qpos_adr + 4] = 0.0
        self.data.qpos[self._rock_qpos_adr + 5] = 0.0
        self.data.qpos[self._rock_qpos_adr + 6] = 0.0

        # Zero rock velocity so it starts at rest
        self.data.qvel[self._rock_qvel_adr: self._rock_qvel_adr + 6] = 0.0

        mujoco.mj_forward(self.model, self.data)

        # Settle rock onto terrain surface, then kill residual velocity
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        self.data.qvel[self._rock_qvel_adr: self._rock_qvel_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._total_height_accumulated = 0.0
        self._prev_rock_height = self.data.xpos[self._rock_id][2]
        self._prev_rock_x = self.data.xpos[self._rock_id][0]

        obs = self._get_obs()
        info = {
            "rock_height": self._prev_rock_height,
            "total_height_accumulated": 0.0,
            "torso_height": self.data.xpos[self._torso_id][2],
        }
        return obs, info

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode is None:
            return None

        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=1080, width=1920)

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = 7.0
        cam.azimuth = 90.0
        cam.elevation = -15.0
        cam.lookat[:] = self.data.xpos[self._torso_id]

        self._renderer.update_scene(self.data, cam)
        pixels = self._renderer.render()

        if self.render_mode == "rgb_array":
            return pixels
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
