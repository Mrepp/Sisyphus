"""Visualize curriculum stages — render the initial scene at each difficulty level.

No trained policy needed. Shows humanoid + rock on terrain at each phase's
slope/mass configuration, running with zero actions for a few seconds.

Usage:
    python scripts/visualize_curriculum.py
    python scripts/visualize_curriculum.py --phases I
    python scripts/visualize_curriculum.py --phases I IV --duration 5.0 --combined
"""

import argparse
import os

import imageio
import mujoco
import numpy as np

from env.sisyphus_env import SisyphusEnv
from render.preview_renderer import PreviewRenderer
from train.curriculum import SCHEDULE


def render_curriculum_stage(
    phase_entry: dict,
    output_path: str,
    duration_seconds: float = 3.0,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
    seed: int = 42,
) -> str:
    """Render one curriculum phase's initial scene to MP4.

    Creates the environment at the given phase's parameters, resets it,
    then steps with zero actions while recording frames.

    Returns:
        Path to the saved MP4 file.
    """
    env = SisyphusEnv(
        slope_deg=phase_entry["slope"],
        rock_mass=phase_entry["mass"],
        infinite_mode=phase_entry["infinite"],
        alive_bonus=phase_entry.get("alive_bonus", 0.0),
        upright_coef=phase_entry.get("upright_coef", 0.0),
        max_steps=999_999,  # won't truncate during short visualization
    )
    obs, info = env.reset(seed=seed)

    # Set up renderer pointing at the env's live model/data
    renderer = PreviewRenderer(env.model, width=width, height=height)
    renderer.data = env.data
    renderer.model = env.model
    renderer._renderer = mujoco.Renderer(env.model, height=height, width=width)

    total_frames = int(duration_seconds * fps)
    physics_dt = env.model.opt.timestep * 5  # 0.015s per env.step()
    frame_dt = 1.0 / fps
    steps_per_frame = max(1, round(frame_dt / physics_dt))

    torso_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    zero_action = np.zeros(env.action_space.shape)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)

    try:
        for frame_idx in range(total_frames):
            lookat = env.data.xpos[torso_id].copy()
            frame = renderer._render_frame(lookat)
            writer.append_data(frame)

            for _ in range(steps_per_frame):
                obs, reward, terminated, truncated, _info = env.step(zero_action)
                if terminated:
                    break
            # After termination, keep stepping physics directly so the
            # scene settles naturally (fallen humanoid, rolling rock).
            if terminated:
                for remaining in range(frame_idx + 1, total_frames):
                    mujoco.mj_step(env.model, env.data, nstep=5 * steps_per_frame)
                    lookat = env.data.xpos[torso_id].copy()
                    frame = renderer._render_frame(lookat)
                    writer.append_data(frame)
                break
    finally:
        writer.close()

    renderer.close()
    env.close()
    return output_path


def combine_videos(video_paths: list[str], output_path: str, fps: int = 30) -> str:
    """Concatenate multiple MP4 files into one."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
    try:
        for path in video_paths:
            reader = imageio.get_reader(path)
            for frame in reader:
                writer.append_data(frame)
            reader.close()
    finally:
        writer.close()
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize curriculum stages — render initial scene at each difficulty level"
    )
    parser.add_argument(
        "--phases", nargs="*", default=None,
        help="Phases to render (e.g., I II III IV). Default: all phases.",
    )
    parser.add_argument(
        "--output-dir", default="renders_preview",
        help="Output directory for MP4s (default: renders_preview)",
    )
    parser.add_argument(
        "--duration", type=float, default=3.0,
        help="Duration of each phase clip in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Output video framerate (default: 30)",
    )
    parser.add_argument(
        "--combined", action="store_true",
        help="Also output a single combined MP4 with all phases",
    )
    parser.add_argument(
        "--width", type=int, default=1920,
        help="Video width in pixels (default: 1920)",
    )
    parser.add_argument(
        "--height", type=int, default=1080,
        help="Video height in pixels (default: 1080)",
    )
    args = parser.parse_args()

    # Filter schedule to requested phases
    if args.phases:
        entries = [e for e in SCHEDULE if e["phase"] in args.phases]
        if not entries:
            valid = [e["phase"] for e in SCHEDULE]
            print(f"No matching phases. Valid phases: {valid}")
            return
    else:
        entries = SCHEDULE

    print("Curriculum Stage Visualizer")
    print("=" * 28)

    video_paths = []
    for entry in entries:
        phase = entry["phase"]
        print(f"\nPhase {phase}:  slope={entry['slope']}deg, "
              f"mass={entry['mass']}kg, infinite={entry['infinite']}")

        output_path = os.path.join(
            args.output_dir, f"curriculum_phase_{phase}.mp4"
        )
        total_frames = int(args.duration * args.fps)
        print(f"  Rendering {args.duration}s ({total_frames} frames)...")

        render_curriculum_stage(
            entry,
            output_path,
            duration_seconds=args.duration,
            fps=args.fps,
            width=args.width,
            height=args.height,
        )
        video_paths.append(output_path)
        print(f"  Saved: {output_path}")

    if args.combined and len(video_paths) > 1:
        combined_path = os.path.join(args.output_dir, "curriculum_all_phases.mp4")
        print(f"\nCreating combined video...")
        combine_videos(video_paths, combined_path, fps=args.fps)
        print(f"  Saved: {combined_path}")

    print(f"\nDone. {len(video_paths)} video(s) saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
