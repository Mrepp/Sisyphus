"""Sisyphus Humanoid Environment — MuJoCo + Gymnasium.

A humanoid pushes a rock up a heightfield incline. Supports curriculum
learning (variable slope/mass) and an infinite-illusion mode for Phase IV.
"""

import os
from collections import deque

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
        obs_hand_dists: bool = True,
        walk_only_mode: bool = False,
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
        self._obs_hand_dists = obs_hand_dists
        self._walk_only_mode = walk_only_mode

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
        self._right_foot_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot")
        self._left_foot_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot")
        self._HAND_RADIUS = 0.04
        self._hfield_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")

        # Cache foot body IDs for COM-over-feet observation
        self._right_foot_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
        self._left_foot_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")

        # Cache back/torso geom IDs for early termination on backward collapse
        self._back_geom_ids = set()
        for name in ("torso1", "uwaist", "head", "butt"):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            self._back_geom_ids.add(gid)

        # Build set of all humanoid geom IDs (for agent-rock contact detection)
        self._humanoid_geom_ids = set()
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and name not in ("terrain_geom", "rock_geom"):
                self._humanoid_geom_ids.add(i)

        # Terrain geometry
        self._hfield_nrow = self.model.hfield_nrow[self._hfield_id]
        self._hfield_ncol = self.model.hfield_ncol[self._hfield_id]
        self._hfield_size = self.model.hfield_size[self._hfield_id].copy()  # (x_half, y_half, z_max, z_base)
        self._terrain_length = self._hfield_size[0] * 2  # full x extent
        # The -x edge of the heightfield in world coordinates
        terrain_geom_x = self.model.geom_pos[self._terrain_geom_id][0]
        self._terrain_x_start = terrain_geom_x - self._hfield_size[0]

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
        self._nu = self.model.nu  # 19 (17 body + 2 ankles)

        # Cache joint IDs for symmetry penalty
        self._left_hip_y_jnt_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_y")
        self._right_hip_y_jnt_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_hip_y")
        self._left_hip_y_qpos_adr = self.model.jnt_qposadr[self._left_hip_y_jnt_id]
        self._right_hip_y_qpos_adr = self.model.jnt_qposadr[self._right_hip_y_jnt_id]

        # Set rock mass
        self._set_rock_mass(rock_mass)

        # Set terrain slope
        self._set_slope(slope_deg)

        # Forward to compute derived quantities
        mujoco.mj_forward(self.model, self.data)

        # State tracking (must be before _get_obs() call)
        self._step_count = 0
        self._prev_rock_height = 0.0
        self._prev_rock_x = 0.0
        self._prev_rock_dist = 0.0
        self._total_height_accumulated = 0.0
        self._reset_x = 1.0

        # Gait tracking
        self._last_sole_foot = None
        self._gait_step_count = 0
        self._last_foot_switch_step = 0
        self._step_intervals = deque(maxlen=10)

        # Action / locomotion smoothness tracking
        self._prev_action = np.zeros(self._nu, dtype=np.float32)
        self._prev_com_z = 1.25
        self._prev_torso_z = 1.25

        # Push attribution tracking
        self._last_rock_contact_step = -10

        # Episode tracking for promotion score
        self._episode_rock_x_start = 0.0
        self._episode_rollback_distance = 0.0
        self._episode_rock_contact_steps = 0
        self._episode_pushed_forward = 0.0

        # --- Spaces ---
        obs_size = self._get_obs().shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_size,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self._nu,), dtype=np.float32,
        )

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
        walk_only_mode: bool = False,
    ):
        self._slope_deg = slope_deg
        self.infinite_mode = infinite_mode
        self._alive_bonus = alive_bonus
        self._upright_coef = upright_coef
        self._forward_push_coef = forward_push_coef
        self._walk_only_mode = False  # ignored — rock always present
        self._set_rock_mass(rock_mass)
        self._set_slope(slope_deg)

    # ------------------------------------------------------------------
    # Terrain
    # ------------------------------------------------------------------
    def _set_slope(self, slope_deg: float):
        """Populate heightfield with a linear incline along +X."""
        nrow = self._hfield_nrow   # rows span the Y-axis
        ncol = self._hfield_ncol   # columns span the X-axis
        z_max = self._hfield_size[2]

        slope_rad = np.radians(slope_deg)
        # MuJoCo hfield: columns map to X-axis, rows map to Y-axis.
        # Column 0 = -x (near), column ncol-1 = +x (far/uphill).
        # We want height increasing with +x (uphill direction).
        col_heights = np.linspace(0, 1, ncol)  # 0 at col 0 (-x, near), 1 at col ncol-1 (+x, uphill)

        # Scale so that max physical height = tan(slope) * terrain_length
        max_h = np.tan(slope_rad) * self._terrain_length
        # Clamp to hfield z_max (normalised values are 0-1 in hfield_data)
        if z_max > 0:
            col_normalized = col_heights * min(max_h / z_max, 1.0)
        else:
            col_normalized = np.zeros(ncol)

        # Tile across rows (same X-profile for every Y row)
        hfield_data = np.tile(col_normalized[None, :], (nrow, 1)).flatten().astype(np.float32)

        # Write into model
        start = self.model.hfield_adr[self._hfield_id]
        self.model.hfield_data[start: start + nrow * ncol] = hfield_data

    def _terrain_height_at_x(self, x: float) -> float:
        """Approximate terrain z at a given world x coordinate."""
        slope_rad = np.radians(self._slope_deg)
        return np.tan(slope_rad) * max(x - self._terrain_x_start, 0.0)

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
    # Foot-terrain contact detection
    # ------------------------------------------------------------------
    def _foot_terrain_contacts(self) -> tuple[bool, bool]:
        """Check if right/left foot geoms are in contact with terrain.

        Returns:
            (right_foot_on_ground, left_foot_on_ground)
        """
        right_on = False
        left_on = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            # Check if one geom is terrain and the other is a foot
            if g1 == self._terrain_geom_id or g2 == self._terrain_geom_id:
                other = g2 if g1 == self._terrain_geom_id else g1
                if other == self._right_foot_geom_id:
                    right_on = True
                elif other == self._left_foot_geom_id:
                    left_on = True
        return right_on, left_on

    # ------------------------------------------------------------------
    # Back-ground contact detection (for early termination)
    # ------------------------------------------------------------------
    def _back_on_ground(self) -> bool:
        """Check if any torso/back/head geom is in contact with terrain."""
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 == self._terrain_geom_id or g2 == self._terrain_geom_id:
                other = g2 if g1 == self._terrain_geom_id else g1
                if other in self._back_geom_ids:
                    return True
        return False

    # ------------------------------------------------------------------
    # Agent-rock contact detection
    # ------------------------------------------------------------------
    def _agent_touching_rock(self) -> bool:
        """Check if any humanoid geom is in contact with the rock geom."""
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 == self._rock_geom_id and g2 in self._humanoid_geom_ids:
                return True
            if g2 == self._rock_geom_id and g1 in self._humanoid_geom_ids:
                return True
        return False

    # ------------------------------------------------------------------
    # Ground reaction forces
    # ------------------------------------------------------------------
    def _foot_ground_forces(self) -> tuple[float, float]:
        """Normal force magnitude on each foot from terrain contacts."""
        right_force = 0.0
        left_force = 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 == self._terrain_geom_id or g2 == self._terrain_geom_id:
                other = g2 if g1 == self._terrain_geom_id else g1
                if other == self._right_foot_geom_id or other == self._left_foot_geom_id:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, i, force)
                    normal_mag = abs(force[0])
                    if other == self._right_foot_geom_id:
                        right_force += normal_mag
                    else:
                        left_force += normal_mag
        return right_force, left_force

    # ------------------------------------------------------------------
    # Agent-rock contact force
    # ------------------------------------------------------------------
    def _agent_rock_contact_force(self) -> float:
        """Total normal force magnitude from agent-rock contacts."""
        total_force = 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            is_rock = (g1 == self._rock_geom_id or g2 == self._rock_geom_id)
            if is_rock:
                other = g2 if g1 == self._rock_geom_id else g1
                if other in self._humanoid_geom_ids:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, i, force)
                    total_force += abs(force[0])
        return total_force

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

        parts = [
            humanoid_qpos,
            humanoid_qvel,
            rock_rel,
            rock_vel,
            torso_height,
            com_vel,
        ]

        # Hand-to-rock surface distances
        if self._obs_hand_dists:
            r_hand_d, l_hand_d = self._hand_rock_distances()
            parts.append(np.array([r_hand_d, l_hand_d]))

        # Foot contacts, agent touching rock, rock Y position
        right_foot_on, left_foot_on = self._foot_terrain_contacts()
        agent_touching = self._agent_touching_rock()
        rock_y = self.data.xpos[self._rock_id][1]
        parts.append(np.array([
            float(right_foot_on),
            float(left_foot_on),
            float(agent_touching),
            rock_y,
        ]))

        # --- Proprioceptive additions ---

        # Actuator force feedback (19 floats) — joint load sensing
        parts.append(self.data.actuator_force.copy())

        # Previous action (19 floats) — temporal motor pattern awareness
        parts.append(self._prev_action.copy())

        # Phase clock (2 floats) — rhythmic locomotion metronome (120-step period ≈ 1.8s gait cycle)
        phase = 2.0 * np.pi * self._step_count / 120.0
        parts.append(np.array([np.sin(phase), np.cos(phase)], dtype=np.float32))

        # Foot ground reaction forces (2 floats) — weight distribution
        # Normalized by 500N (typical standing GRF ~400-800N per foot)
        right_grf, left_grf = self._foot_ground_forces()
        parts.append(np.array([right_grf / 500.0, left_grf / 500.0], dtype=np.float32))

        # Rock contact force magnitude (1 float) — push intensity
        # Normalized by 500N to keep O(1) scale
        parts.append(np.array([self._agent_rock_contact_force() / 500.0], dtype=np.float32))

        # Terrain slope (1 float) — slope-conditioned policy
        parts.append(np.array([self._slope_deg / 15.0], dtype=np.float32))

        # Torso gravity vector in body frame (3 floats) — tilt awareness
        torso_xmat = self.data.xmat[self._torso_id].reshape(3, 3)
        gravity_body = torso_xmat[2, :]  # world z-axis in body frame
        parts.append(gravity_body.astype(np.float32))

        # COM-over-feet offset in torso local frame (2 floats) — balance proprioception
        # Positive X = COM ahead of feet (stable for pushing), negative = behind (unstable)
        com_xy = self.data.subtree_com[0][:2]
        rf_xy = self.data.xpos[self._right_foot_body_id][:2]
        lf_xy = self.data.xpos[self._left_foot_body_id][:2]
        feet_mid = 0.5 * (rf_xy + lf_xy)
        offset_world = com_xy - feet_mid
        com_offset_local = np.array([
            torso_xmat[0, 0] * offset_world[0] + torso_xmat[1, 0] * offset_world[1],
            torso_xmat[0, 1] * offset_world[0] + torso_xmat[1, 1] * offset_world[1],
        ], dtype=np.float32)
        parts.append(com_offset_local)

        obs = np.concatenate(parts).astype(np.float32)
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

        # Rock X movement
        rock_x = rock_pos[0]
        delta_x_rock = rock_x - self._prev_rock_x
        self._prev_rock_x = rock_x

        # Rock Y position (lateral)
        rock_y = rock_pos[1]

        # Episode rollback tracking
        if delta_x_rock < 0:
            self._episode_rollback_distance += abs(delta_x_rock)

        # --- Contact detection ---
        right_foot_on, left_foot_on = self._foot_terrain_contacts()
        agent_touching_rock = self._agent_touching_rock()

        # Update rock contact tracking for push attribution
        if agent_touching_rock:
            self._last_rock_contact_step = self._step_count
            self._episode_rock_contact_steps += 1

        # --- Contact intensity & push gating ---
        rock_contact_force = self._agent_rock_contact_force()
        contact_intensity = np.clip(
            rock_contact_force / 200.0, 0.0, 1.0
        )
        # Soft push gate: any contact counts (binary or force-based)
        push_gate = max(
            contact_intensity,
            0.5 * float(agent_touching_rock),
        )

        # Suppress locomotion style priors during active pushing
        style_damping = 1.0 - push_gate

        # Accumulate agent-pushed forward displacement
        if push_gate > 0 and delta_x_rock > 0:
            if torso_pos[0] <= rock_x:
                self._episode_pushed_forward += delta_x_rock

        # --- Posture ---
        torso_xmat = self.data.xmat[self._torso_id].reshape(3, 3)
        torso_up_dot = torso_xmat[2, 2]
        # Upright gate: sqrt preserves gradient at moderate tilt, zeroes at horizontal
        # up_dot=1.0→1.0, up_dot=0.8→0.89, up_dot=0.5→0.71, up_dot=0.0→0.0
        upright_gate = max(0.0, torso_up_dot) ** 0.5

        # Backward lean detection: Z component of torso forward axis
        # Positive = torso leans backward (forward axis points upward)
        torso_forward_z = torso_xmat[2, 0]

        # COM-over-feet for balance reward
        com_xy = self.data.subtree_com[0][:2]
        rf_xy = self.data.xpos[self._right_foot_body_id][:2]
        lf_xy = self.data.xpos[self._left_foot_body_id][:2]
        feet_mid_xy = 0.5 * (rf_xy + lf_xy)
        com_ahead = com_xy[0] - feet_mid_xy[0]  # positive = COM ahead of feet

        # --- Rewards ---

        # Forward-push reward (push-gated, dominant signal)
        forward_reward = 15.0 * max(delta_x_rock, 0) * push_gate

        # Height reward (push-gated, for uphill phases)
        height_reward = 50.0 * max(delta_h_rock, 0) * push_gate

        # Rock rollback penalty
        rock_rollback_penalty = (
            5.0 * max(-delta_x_rock, 0)
            if self._slope_deg > 0 else 0.0
        )

        # Lateral drift penalty (capped to avoid drowning posture rewards)
        lateral_penalty = min(2.0 * abs(rock_y), 3.0)

        # Cost of transport (replaces torque penalty)
        qvel = self.data.qvel
        joint_vel = qvel[self._root_qvel_adr + 6:
                         self._rock_qvel_adr]
        mechanical_power = np.sum(
            np.abs(self.data.actuator_force * joint_vel)
        )
        com_vel_x = self.data.subtree_linvel[0][0]
        speed = max(abs(com_vel_x), 0.1)
        cot_penalty = 0.002 * mechanical_power / speed

        # Action smoothness penalty
        action_delta = action - self._prev_action
        smoothness_penalty = 0.005 * np.sum(action_delta ** 2) * style_damping

        # Unified height reward (replaces alive_bonus + fall_penalty + smooth_height)
        _HEIGHT_TARGET = 1.25   # standing height
        _HEIGHT_ZERO = 0.6     # reward crosses zero here
        _HEIGHT_FLOOR = 0.2    # penalty saturates here
        if torso_z >= _HEIGHT_TARGET:
            height_reward_posture = 2.0
        elif torso_z >= _HEIGHT_ZERO:
            frac = (torso_z - _HEIGHT_ZERO) / (_HEIGHT_TARGET - _HEIGHT_ZERO)
            height_reward_posture = 2.0 * frac
        elif torso_z >= _HEIGHT_FLOOR:
            frac = (_HEIGHT_ZERO - torso_z) / (_HEIGHT_ZERO - _HEIGHT_FLOOR)
            height_reward_posture = -2.0 * frac
        else:
            height_reward_posture = -2.0
        height_reward_posture *= min(self._upright_coef, 2.0)

        # Getting-up reward: continuous height-proportional + velocity bonus
        # Fades out above z=1.0 to avoid penalizing lower pushing postures
        _GETUP_CEILING = 1.0
        torso_z_delta = torso_z - self._prev_torso_z
        if torso_z < _GETUP_CEILING:
            height_frac = max(
                0.0,
                (torso_z - _HEIGHT_FLOOR)
                / (_GETUP_CEILING - _HEIGHT_FLOOR),
            )
            getup_reward = (
                2.0 * height_frac
                + 5.0 * max(torso_z_delta, 0.0)
            )
        else:
            getup_reward = 0.0
        self._prev_torso_z = torso_z

        # Approach reward: exponential proximity shaping (always active)
        rock_dist = np.linalg.norm(rock_pos[:2] - torso_pos[:2])
        approach_reward = 3.0 * np.exp(-rock_dist) * upright_gate
        self._prev_rock_dist = rock_dist

        # Small alive bonus (curriculum-decayed)
        alive_bonus = self._alive_bonus

        # --- Walking rewards ---

        # COM forward velocity reward (reduced, fades near rock)
        proximity_damping = min(1.0, rock_dist / 2.0)
        walk_reward = (
            1.5 * min(max(com_vel_x, 0.0), 1.5)
            * upright_gate * proximity_damping
        )

        # Cadence reward (replaces simple gait reward)
        cadence_reward = 0.0
        if right_foot_on and not left_foot_on:
            current_sole = "right"
        elif left_foot_on and not right_foot_on:
            current_sole = "left"
        else:
            current_sole = None

        if (current_sole is not None
                and current_sole != self._last_sole_foot):
            interval = (
                self._step_count - self._last_foot_switch_step
            )
            self._step_intervals.append(interval)
            self._last_foot_switch_step = self._step_count
            self._last_sole_foot = current_sole
            self._gait_step_count += 1

            # Target ~40 steps (~0.6s at 67Hz) between switches
            TARGET_INTERVAL = 40
            INTERVAL_TOLERANCE = 25
            interval_error = abs(interval - TARGET_INTERVAL)
            interval_bonus = max(
                0.0, 1.0 - interval_error / INTERVAL_TOLERANCE
            )

            # Regularity: consistent stride intervals
            regularity_bonus = 0.0
            if len(self._step_intervals) >= 4:
                intervals = np.array(self._step_intervals)
                cv = np.std(intervals) / (
                    np.mean(intervals) + 1e-8
                )
                regularity_bonus = max(0.0, 1.0 - cv)

            vel_scale = np.clip(com_vel_x / 0.6, 0.0, 1.0)
            cadence_reward = 1.0 * vel_scale * (
                0.5 * interval_bonus + 0.5 * regularity_bonus
            ) * style_damping

        # Continuous per-step cadence reward (supplements event-based)
        # Phase from the gait clock: sin(phase) > 0 → expect right foot down
        phase_for_cadence = 2.0 * np.pi * self._step_count / 120.0
        expected_right = float(np.sin(phase_for_cadence) > 0)
        expected_left = 1.0 - expected_right
        phase_match = (
            expected_right * float(right_foot_on)
            + expected_left * float(left_foot_on)
        )
        cadence_continuous = 0.15 * phase_match * upright_gate * style_damping

        # Bilateral symmetry penalty: penalize asymmetric hip angles
        left_hip_y_pos = self.data.qpos[self._left_hip_y_qpos_adr]
        right_hip_y_pos = self.data.qpos[self._right_hip_y_qpos_adr]
        symmetry_penalty = 0.05 * abs(left_hip_y_pos - right_hip_y_pos) * style_damping

        # Stance reward
        any_foot_on = float(right_foot_on or left_foot_on)
        stance_reward = 0.5 * any_foot_on * upright_gate

        # COM height stability reward
        com_z = self.data.subtree_com[0][2]
        com_z_change = abs(com_z - self._prev_com_z)
        com_stability_reward = 0.3 * max(
            0.0, 1.0 - com_z_change / 0.05
        )
        self._prev_com_z = com_z

        # Balance reward: standing upright with feet on ground (lowered gate for approach crouch)
        if torso_z >= 0.7 and any_foot_on:
            balance_reward = 1.5 * max(0.0, torso_up_dot)
        else:
            balance_reward = 0.0

        # Idle penalty (suppressed when bracing against rock)
        idle_penalty = (
            -0.1
            if (com_vel_x < 0.05 and self._step_count > 50
                and not agent_touching_rock)
            else 0.0
        )

        # Touch bonus: binary reward for any body-rock contact
        touch_bonus = 1.0 * float(agent_touching_rock) * upright_gate

        # Contact reward: scales with applied force
        contact_reward = (
            2.0 * contact_intensity * upright_gate + touch_bonus
        )

        # Hand proximity reward: encourage hands near rock
        right_dist, left_dist = self._hand_rock_distances()
        min_hand_dist = min(right_dist, left_dist)
        hand_near_reward = (
            1.5 * np.exp(-2.0 * min_hand_dist) * upright_gate
        )

        # Body lean reward: torso oriented toward rock
        # Gated on feet on ground and not leaning backward
        lean_reward = 0.0
        if rock_dist < 2.0 and any_foot_on and torso_up_dot > 0.5 and torso_forward_z <= 0.1:
            torso_fwd = torso_xmat[:, 0]
            rock_dir = rock_pos[:2] - torso_pos[:2]
            rock_dir_n = rock_dir / (
                np.linalg.norm(rock_dir) + 1e-8
            )
            lean_dot = (
                torso_fwd[0] * rock_dir_n[0]
                + torso_fwd[1] * rock_dir_n[1]
            )
            lean_reward = 1.5 * max(lean_dot, 0.0)

        # Backward lean penalty: penalize torso tilting backward
        # torso_forward_z > 0.1 means backward lean beyond dead zone
        # Not damped by style_damping — stability signal, not style
        backward_lean_penalty = 0.0
        if torso_forward_z > 0.1:
            backward_lean_penalty = 2.0 * (torso_forward_z - 0.1)

        # Forward lean penalty: penalize excessive forward tilt
        # Larger dead zone (0.3 vs 0.1) since forward lean is useful for pushing
        # Not damped — past ~17 deg forward is a stability problem
        forward_lean_penalty = 0.0
        if torso_forward_z < -0.3:
            forward_lean_penalty = 2.0 * (-torso_forward_z - 0.3)

        # COM-over-feet reward: reward COM ahead of feet, penalize behind
        # Not damped by style_damping — stability signal, not style
        if any_foot_on and torso_z >= 0.7:
            com_balance_reward = 0.5 * np.clip(com_ahead, -1.0, 0.3)
        else:
            com_balance_reward = 0.0

        # Update previous action for next step
        self._prev_action = action.copy()

        reward = (
            # Tier 1: Posture
            height_reward_posture + alive_bonus + getup_reward
            # Tier 2: Locomotion
            + walk_reward + cadence_reward + cadence_continuous
            + stance_reward + balance_reward + com_stability_reward
            + com_balance_reward
            # Tier 3: Rock interaction (dominant)
            + approach_reward + contact_reward
            + forward_reward + height_reward
            + hand_near_reward + lean_reward
            # Tier 4: Penalties
            - cot_penalty - smoothness_penalty - symmetry_penalty
            - lateral_penalty - rock_rollback_penalty
            - backward_lean_penalty - forward_lean_penalty
            + idle_penalty
        )

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

            # Reset rock tracking to avoid false delta_x / rollback spike
            self._prev_rock_x = self.data.xpos[self._rock_id][0]
            self._prev_rock_dist = np.linalg.norm(
                self.data.xpos[self._rock_id][:2] - self.data.xpos[self._torso_id][:2]
            )

        # Early termination: back/torso contacts ground after grace period
        back_contact = self._back_on_ground()
        terminated = back_contact and self._step_count > 50
        truncated = self._step_count >= self.max_steps

        # Compute episode metrics for promotion score
        rock_delta_x_total = rock_pos[0] - self._episode_rock_x_start

        info = {
            "rock_height": current_rock_height,
            "total_height_accumulated": self._total_height_accumulated,
            "torso_height": torso_z,
            "torso_up_dot": torso_up_dot,
            "alive_bonus": alive_bonus,
            "height_reward_posture": height_reward_posture,
            "getup_reward": getup_reward,
            "forward_reward": forward_reward,
            "height_reward": height_reward,
            "contact_reward": contact_reward,
            "touch_bonus": touch_bonus,
            "contact_intensity": contact_intensity,
            "hand_near_reward": hand_near_reward,
            "right_hand_dist": right_dist,
            "left_hand_dist": left_dist,
            "step_count": self._step_count,
            "rock_distance": rock_dist,
            "com_vel_x": com_vel_x,
            "rock_delta_x": delta_x_rock,
            "walk_reward": walk_reward,
            "approach_reward": approach_reward,
            "idle_penalty": idle_penalty,
            "cadence_reward": cadence_reward,
            "stance_reward": stance_reward,
            "lateral_penalty": lateral_penalty,
            "rock_rollback_penalty": rock_rollback_penalty,
            "push_gate": push_gate,
            "rock_contact_force": rock_contact_force,
            "rock_delta_x_total": rock_delta_x_total,
            "gait_step_count": self._gait_step_count,
            "right_foot_on": float(right_foot_on),
            "left_foot_on": float(left_foot_on),
            "agent_touching_rock": float(agent_touching_rock),
            "rock_y": rock_y,
            "pushed_forward": self._episode_pushed_forward,
            "cot_penalty": cot_penalty,
            "smoothness_penalty": smoothness_penalty,
            "com_stability_reward": com_stability_reward,
            "lean_reward": lean_reward,
            "cadence_continuous": cadence_continuous,
            "symmetry_penalty": symmetry_penalty,
            "balance_reward": balance_reward,
            "backward_lean_penalty": backward_lean_penalty,
            "forward_lean_penalty": forward_lean_penalty,
            "com_balance_reward": com_balance_reward,
            "back_on_ground": float(back_contact),
            "style_damping": style_damping,
        }

        # Add promotion_score on terminal step — requires active pushing
        if terminated or truncated:
            contact_fraction = (self._episode_rock_contact_steps / max(self._step_count, 1))
            # Only credit rock displacement from active pushing, require ≥5% contact
            if contact_fraction >= 0.05:
                promotion_score = (self._episode_pushed_forward
                                   - 2.0 * abs(rock_y)
                                   - self._episode_rollback_distance)
            else:
                promotion_score = 0.0  # no credit if agent never actively pushed
            info["promotion_score"] = promotion_score
            info["contact_fraction"] = contact_fraction
            info["pushed_forward"] = self._episode_pushed_forward

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Explicit standing pose — never rely on key_qpos being correct
        self.data.qpos[:] = 0.0
        self.data.qpos[self._root_qpos_adr + 2] = 1.4      # z height
        self.data.qpos[self._root_qpos_adr + 3] = 1.0      # quat w (identity)
        # Small random joint perturbation for exploration diversity
        n_joints = self._rock_qpos_adr - (self._root_qpos_adr + 7)
        self.data.qpos[self._root_qpos_adr + 7: self._rock_qpos_adr] += (
            self.np_random.normal(0, 0.005, size=n_joints)
        )

        # Offset humanoid z by terrain height at its x position
        root_x = self.data.qpos[self._root_qpos_adr]
        self.data.qpos[self._root_qpos_adr + 2] += self._terrain_height_at_x(root_x)

        # Rock always placed 2-3m ahead for approach → contact → push learning
        rock_x = self.np_random.uniform(2.0, 3.0)

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

        # Randomize rock friction for robustness
        rock_friction_mu = self.np_random.uniform(1.0, 2.0)
        self.model.geom_friction[self._rock_geom_id][0] = rock_friction_mu

        mujoco.mj_forward(self.model, self.data)

        # Settle rock onto terrain surface, then kill ALL residual velocity
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        self.data.qvel[self._rock_qvel_adr: self._rock_qvel_adr + 6] = 0.0
        # Zero humanoid velocity too — settling steps cause it to start mid-fall
        self.data.qvel[self._root_qvel_adr: self._rock_qvel_adr] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._total_height_accumulated = 0.0
        self._prev_rock_height = self.data.xpos[self._rock_id][2]
        self._prev_rock_x = self.data.xpos[self._rock_id][0]
        torso_pos = self.data.xpos[self._torso_id]
        rock_pos = self.data.xpos[self._rock_id]
        self._prev_rock_dist = np.linalg.norm(rock_pos[:2] - torso_pos[:2])

        # Reset gait tracking
        self._last_sole_foot = None
        self._gait_step_count = 0
        self._last_foot_switch_step = 0
        self._step_intervals.clear()

        # Reset action / locomotion smoothness tracking
        self._prev_action[:] = 0.0
        self._prev_com_z = self.data.subtree_com[0][2]
        self._prev_torso_z = self.data.xpos[self._torso_id][2]

        # Reset push attribution tracking
        self._last_rock_contact_step = -10

        # Reset episode tracking
        self._episode_rock_x_start = self.data.xpos[self._rock_id][0]
        self._episode_rollback_distance = 0.0
        self._episode_rock_contact_steps = 0
        self._episode_pushed_forward = 0.0

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
