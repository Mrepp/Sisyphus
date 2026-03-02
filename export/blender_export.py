"""Blender export pipeline — converts trajectory .npz to Blender-friendly formats.

Outputs:
  - joint_animation.json: per-frame joint angles + root position/quaternion
  - rock_trajectory.json: per-frame rock position + quaternion
  - terrain_heightfield.npz: raw heightfield data for mesh reconstruction
  - scene_metadata.json: slope, mass, checkpoint, coordinate system info

Compatible with danieldugas/blender_mujoco for armature import.
"""

import os
import json
import numpy as np
import mujoco

from logging_utils.trajectory_logger import TrajectoryLogger


# MuJoCo humanoid joint names in actuator order
JOINT_NAMES = [
    "abdomen_y", "abdomen_z", "abdomen_x",
    "right_hip_x", "right_hip_z", "right_hip_y", "right_knee",
    "left_hip_x", "left_hip_z", "left_hip_y", "left_knee",
    "right_shoulder1", "right_shoulder2", "right_elbow",
    "left_shoulder1", "left_shoulder2", "left_elbow",
]


def resample_trajectory(data: dict, target_fps: float = 60.0) -> dict:
    """Resample trajectory to a target FPS using linear interpolation.

    Args:
        data: Trajectory dict from TrajectoryLogger.load().
        target_fps: Desired output framerate.

    Returns:
        Resampled trajectory dict with the same keys.
    """
    metadata = data.get("metadata", {})
    source_fps = metadata.get("fps", 66.67)  # default: 1/(0.003*5)

    T_source = len(data["qpos"])
    duration = T_source / source_fps
    T_target = int(duration * target_fps)

    source_times = np.linspace(0, duration, T_source)
    target_times = np.linspace(0, duration, T_target)

    result = {}
    for key in ["qpos", "qvel", "rock_pos", "torques", "com", "rewards"]:
        if key in data:
            arr = data[key]
            if arr.ndim == 1:
                result[key] = np.interp(target_times, source_times, arr)
            else:
                resampled = np.zeros((T_target, arr.shape[1]))
                for col in range(arr.shape[1]):
                    resampled[:, col] = np.interp(target_times, source_times, arr[:, col])
                result[key] = resampled

    # Contact forces: take nearest neighbour (interpolation not meaningful)
    if "contact_forces" in data:
        indices = np.searchsorted(source_times, target_times, side="right") - 1
        indices = np.clip(indices, 0, T_source - 1)
        result["contact_forces"] = data["contact_forces"][indices]

    result["metadata"] = metadata
    return result


def _extract_joint_angles(qpos: np.ndarray, model: mujoco.MjModel) -> dict:
    """Extract named joint angles from a single qpos vector.

    Returns dict mapping joint_name → angle_in_degrees.
    """
    angles = {}
    for name in JOINT_NAMES:
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jnt_id < 0:
            continue
        adr = model.jnt_qposadr[jnt_id]
        # Hinge joints have 1 qpos value (in radians)
        angles[name] = float(np.degrees(qpos[adr]))
    return angles


def _extract_root_pose(qpos: np.ndarray, model: mujoco.MjModel) -> tuple:
    """Extract root body position and quaternion from qpos.

    Returns (position [3], quaternion [4] as wxyz).
    """
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    adr = model.jnt_qposadr[root_id]
    pos = qpos[adr: adr + 3].tolist()
    quat = qpos[adr + 3: adr + 7].tolist()  # MuJoCo: w, x, y, z
    return pos, quat


def _extract_rock_pose(qpos: np.ndarray, model: mujoco.MjModel) -> tuple:
    """Extract rock position and quaternion from qpos."""
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rock_joint")
    adr = model.jnt_qposadr[jnt_id]
    pos = qpos[adr: adr + 3].tolist()
    quat = qpos[adr + 3: adr + 7].tolist()
    return pos, quat


def export_for_blender(
    trajectory_path: str,
    output_dir: str,
    model_path: str | None = None,
    target_fps: float = 60.0,
):
    """Export a trajectory .npz as Blender-friendly files.

    Args:
        trajectory_path: Path to .npz trajectory.
        output_dir: Directory for output files.
        model_path: Path to MuJoCo XML (auto-detected if None).
        target_fps: Output animation framerate.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load trajectory
    traj = TrajectoryLogger.load(trajectory_path)

    # Resample to target FPS
    traj = resample_trajectory(traj, target_fps=target_fps)

    # Load MuJoCo model for joint info
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "sisyphus_humanoid.xml")
    model = mujoco.MjModel.from_xml_path(os.path.normpath(model_path))

    qpos_seq = traj["qpos"]
    T = len(qpos_seq)
    dt = 1.0 / target_fps

    # --- 1. Joint animation ---
    joint_frames = []
    root_positions = []
    root_quaternions = []

    for t in range(T):
        qpos = qpos_seq[t]
        angles = _extract_joint_angles(qpos, model)
        pos, quat = _extract_root_pose(qpos, model)
        joint_frames.append({"time": round(t * dt, 6), "joints": angles})
        root_positions.append(pos)
        root_quaternions.append(quat)

    joint_animation = {
        "fps": target_fps,
        "total_frames": T,
        "duration": round(T * dt, 4),
        "joint_names": JOINT_NAMES,
        "frames": joint_frames,
        "root_position": root_positions,
        "root_quaternion": root_quaternions,
        "coordinate_system": "MuJoCo (z-up, right-handed)",
        "angle_unit": "degrees",
        "quaternion_format": "wxyz",
    }

    with open(os.path.join(output_dir, "joint_animation.json"), "w") as f:
        json.dump(joint_animation, f, indent=2)

    # --- 2. Rock trajectory ---
    rock_frames = []
    for t in range(T):
        qpos = qpos_seq[t]
        pos, quat = _extract_rock_pose(qpos, model)
        # Apply height offset from trajectory rock_pos if available
        if "rock_pos" in traj:
            pos = traj["rock_pos"][t].tolist()
        rock_frames.append({
            "time": round(t * dt, 6),
            "position": pos,
            "quaternion": quat,
        })

    rock_trajectory = {
        "fps": target_fps,
        "total_frames": T,
        "frames": rock_frames,
        "coordinate_system": "MuJoCo (z-up, right-handed)",
        "quaternion_format": "wxyz",
    }

    with open(os.path.join(output_dir, "rock_trajectory.json"), "w") as f:
        json.dump(rock_trajectory, f, indent=2)

    # --- 3. Terrain heightfield ---
    hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")
    nrow = model.hfield_nrow[hfield_id]
    ncol = model.hfield_ncol[hfield_id]
    size = model.hfield_size[hfield_id].copy()
    adr = model.hfield_adr[hfield_id]
    hfield_data = model.hfield_data[adr: adr + nrow * ncol].reshape(nrow, ncol).copy()

    np.savez(
        os.path.join(output_dir, "terrain_heightfield.npz"),
        data=hfield_data,
        size=size,
        nrow=nrow,
        ncol=ncol,
    )

    # --- 4. Scene metadata ---
    metadata = traj.get("metadata", {})
    scene_meta = {
        "slope_deg": metadata.get("slope", 0.0),
        "rock_mass_kg": metadata.get("mass", 8.0),
        "checkpoint": metadata.get("checkpoint", -1),
        "total_training_steps": metadata.get("total_steps", -1),
        "source_fps": metadata.get("fps", 66.67),
        "export_fps": target_fps,
        "total_frames": T,
        "duration_seconds": round(T * dt, 4),
        "coordinate_system": "MuJoCo: z-up, right-handed, meters",
        "quaternion_format": "wxyz (scalar first)",
        "blender_notes": {
            "import_tool": "danieldugas/blender_mujoco",
            "z_up_to_blender": "Blender also uses z-up by default; no axis swap needed.",
            "joint_angles": "Degrees, relative to parent body frame.",
            "root_motion": "root_position and root_quaternion define world-space torso pose.",
        },
    }

    with open(os.path.join(output_dir, "scene_metadata.json"), "w") as f:
        json.dump(scene_meta, f, indent=2)

    return output_dir
