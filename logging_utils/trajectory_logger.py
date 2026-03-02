"""Trajectory logger — records simulation state for replay and export.

Saves the universal .npz format consumed by both the preview renderer
and the Blender export pipeline.
"""

import os
import json
import numpy as np
import mujoco


class TrajectoryLogger:
    def __init__(self, save_dir: str = "replays"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.reset()

    def reset(self):
        """Clear all buffers for a new episode."""
        self._qpos = []
        self._qvel = []
        self._torques = []
        self._rock_pos = []
        self._com = []
        self._rewards = []
        self._contact_forces = []

    def record_step(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        action: np.ndarray,
        reward: float,
        rock_body_id: int,
        height_offset: float = 0.0,
        max_contacts: int = 10,
    ):
        """Record one timestep of simulation data.

        Args:
            model: MuJoCo model.
            data: MuJoCo data (post-step).
            action: Applied control action.
            reward: Scalar reward for this step.
            rock_body_id: Body ID of the rock.
            height_offset: Accumulated height offset from infinite illusion teleports.
            max_contacts: Max contacts to record per step.
        """
        self._qpos.append(data.qpos.copy())
        self._qvel.append(data.qvel.copy())
        self._torques.append(action.copy())

        # Rock world position, adjusted for accumulated height
        rock_pos = data.xpos[rock_body_id].copy()
        rock_pos[2] += height_offset
        self._rock_pos.append(rock_pos)

        # Whole-model center of mass
        self._com.append(data.subtree_com[0].copy())

        self._rewards.append(reward)

        # Contact forces: collect up to max_contacts
        contacts = np.zeros((max_contacts, 6), dtype=np.float64)
        n_active = min(data.ncon, max_contacts)
        for i in range(n_active):
            c = data.contact[i]
            # 6D contact force: 3 normal + 3 tangent
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            contacts[i] = force
        self._contact_forces.append(contacts)

    def save_episode(
        self,
        episode_id: int,
        checkpoint_id: int,
        metadata: dict | None = None,
    ) -> str:
        """Save recorded episode as compressed .npz.

        Returns:
            Path to the saved file.
        """
        if len(self._qpos) == 0:
            raise ValueError("No data recorded — call record_step first.")

        metadata = metadata or {}
        filename = f"episode_{episode_id}_checkpoint_{checkpoint_id}.npz"
        filepath = os.path.join(self.save_dir, filename)

        # Convert metadata to JSON string for safe npz storage
        metadata_str = json.dumps(metadata)

        np.savez_compressed(
            filepath,
            qpos=np.array(self._qpos),
            qvel=np.array(self._qvel),
            rock_pos=np.array(self._rock_pos),
            torques=np.array(self._torques),
            com=np.array(self._com),
            rewards=np.array(self._rewards),
            contact_forces=np.array(self._contact_forces),
            metadata=metadata_str,
        )
        return filepath

    @staticmethod
    def load(filepath: str) -> dict:
        """Load a trajectory .npz and return its contents as a dict."""
        data = np.load(filepath, allow_pickle=True)
        result = {key: data[key] for key in data.files}
        # Parse metadata JSON
        if "metadata" in result:
            try:
                result["metadata"] = json.loads(str(result["metadata"]))
            except (json.JSONDecodeError, TypeError):
                pass
        return result
