"""Load an SB3 checkpoint, run evaluation episodes, render MP4s and save trajectories.

Usage:
    python scripts/render_checkpoint.py --checkpoint logs/checkpoints/sisyphus_ppo_500000_steps \
        --output renders_preview/test.mp4 --episodes 3
"""

import argparse
import os

import numpy as np
from stable_baselines3 import PPO

from env.sisyphus_env import SisyphusEnv
from logging_utils.trajectory_logger import TrajectoryLogger
from render.preview_renderer import PreviewRenderer


def evaluate_and_render(
    checkpoint_path: str,
    output_dir: str = "renders_preview",
    replay_dir: str = "replays",
    num_episodes: int = 1,
    slope: float = 0.0,
    rock_mass: float = 40.0,
    max_steps: int = 1000,
):
    env = SisyphusEnv(slope_deg=slope, rock_mass=rock_mass, max_steps=max_steps)
    model = PPO.load(checkpoint_path)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(replay_dir, exist_ok=True)

    renderer = PreviewRenderer(env.model, width=1920, height=1080)
    checkpoint_id = os.path.basename(checkpoint_path)

    for ep in range(num_episodes):
        print(f"Episode {ep + 1}/{num_episodes}")

        # Run episode with trajectory logging
        traj_logger = TrajectoryLogger(save_dir=replay_dir)
        obs, info = env.reset()

        total_reward = 0.0
        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            traj_logger.record_step(
                model=env.model,
                data=env.data,
                action=action,
                reward=reward,
                rock_body_id=env._rock_id,
                height_offset=info.get("total_height_accumulated", 0.0),
            )

            if terminated or truncated:
                break

        print(f"  Total reward: {total_reward:.2f}, Steps: {step + 1}")

        # Save trajectory
        metadata = {
            "slope": slope,
            "mass": rock_mass,
            "checkpoint": checkpoint_id,
            "timestep": float(env.model.opt.timestep),
            "fps": 1.0 / (env.model.opt.timestep * 5),
        }
        traj_path = traj_logger.save_episode(ep, checkpoint_id=0, metadata=metadata)
        print(f"  Saved trajectory: {traj_path}")

        # Render MP4
        mp4_path = os.path.join(output_dir, f"eval_{checkpoint_id}_ep{ep}.mp4")
        renderer.render_trajectory(traj_path, mp4_path, fps=30)
        print(f"  Saved preview: {mp4_path}")

    renderer.close()
    env.close()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a checkpoint evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to SB3 checkpoint")
    parser.add_argument("--output", default="renders_preview", help="Output directory for MP4s")
    parser.add_argument("--replays", default="replays", help="Output directory for trajectories")
    parser.add_argument("--episodes", type=int, default=1, help="Number of evaluation episodes")
    parser.add_argument("--slope", type=float, default=0.0, help="Terrain slope (degrees)")
    parser.add_argument("--mass", type=float, default=8.0, help="Rock mass (kg)")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    args = parser.parse_args()

    evaluate_and_render(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        replay_dir=args.replays,
        num_episodes=args.episodes,
        slope=args.slope,
        rock_mass=args.mass,
        max_steps=args.max_steps,
    )
