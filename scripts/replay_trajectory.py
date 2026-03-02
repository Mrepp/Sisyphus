"""Replay a saved trajectory .npz in MuJoCo's interactive viewer.

Usage:
    python scripts/replay_trajectory.py replays/episode_0_checkpoint_5.npz
"""

import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

from logging_utils.trajectory_logger import TrajectoryLogger


def replay(trajectory_path: str, model_path: str = "models/sisyphus_humanoid.xml", speed: float = 1.0):
    traj = TrajectoryLogger.load(trajectory_path)
    qpos_seq = traj["qpos"]
    metadata = traj.get("metadata", {})
    T = len(qpos_seq)

    source_fps = metadata.get("fps", 66.67)
    dt = 1.0 / (source_fps * speed)

    print(f"Loaded trajectory: {T} frames, {source_fps:.1f} FPS, {T / source_fps:.1f}s")
    print(f"Metadata: {metadata}")

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for t in range(T):
            if not viewer.is_running():
                break

            data.qpos[:] = qpos_seq[t]
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(dt)

    print("Replay complete.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/replay_trajectory.py <trajectory.npz> [model.xml] [speed]")
        sys.exit(1)

    traj_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "models/sisyphus_humanoid.xml"
    speed = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

    replay(traj_path, model_path, speed)
