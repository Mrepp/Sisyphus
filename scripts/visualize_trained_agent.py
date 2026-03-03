"""Visualize a trained agent at its latest curriculum stage.

Downloads the latest PPO checkpoint from a Google Drive shared folder,
determines which curriculum phase that checkpoint reached, and renders
the trained policy pushing the rock at the correct slope/mass settings.

Optionally renders a zero-action baseline for comparison.

Usage:
    python scripts/visualize_trained_agent.py
    python scripts/visualize_trained_agent.py --no-baseline --max-steps 2000
    python scripts/visualize_trained_agent.py --checkpoint /local/path/to/sisyphus_ppo_5000000_steps
"""

import argparse
import glob
import os
import re
import sys

# Ensure project root is on the path when running as a script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from stable_baselines3 import PPO

from env.sisyphus_env import SisyphusEnv
from render.preview_renderer import PreviewRenderer
from train.curriculum import SCHEDULE, CurriculumManager


DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1bq_jR1oP4aUvxDi1A8gMRFvlgNIQKEaZ"
LOCAL_CHECKPOINT_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "checkpoints"
)


class ZeroPolicy:
    """Mimics the SB3 model.predict() interface but returns zero actions."""

    def __init__(self, action_shape):
        self._zeros = np.zeros(action_shape)

    def predict(self, obs, deterministic=True):
        return self._zeros, None


def download_latest_checkpoint(folder_url: str, local_dir: str) -> tuple[str, int]:
    """List a Google Drive shared folder, find the highest-step checkpoint,
    and download only that file.

    Returns:
        (checkpoint_path_without_extension, total_steps)
    """
    import gdown

    os.makedirs(local_dir, exist_ok=True)

    # List folder contents without downloading
    print(f"Scanning Google Drive folder for checkpoints...")
    file_list = gdown.download_folder(folder_url, skip_download=True, quiet=True)

    if not file_list:
        raise FileNotFoundError(
            f"No files found in Google Drive folder. "
            f"Train a model first using sisyphus_train.ipynb."
        )

    # Parse step counts and find the latest
    step_re = re.compile(r"sisyphus_ppo_(\d+)_steps\.zip")
    parsed = []
    final_entry = None
    for entry in file_list:
        name = os.path.basename(entry.path)
        if name == "sisyphus_ppo_final.zip":
            final_entry = entry
            continue
        match = step_re.search(name)
        if match:
            parsed.append((int(match.group(1)), entry))

    # Prefer _final, otherwise highest step count
    if final_entry:
        best_entry = final_entry
        total_steps = SCHEDULE[-1]["end_step"]
    elif parsed:
        parsed.sort(key=lambda x: x[0], reverse=True)
        total_steps, best_entry = parsed[0]
    else:
        raise FileNotFoundError(
            "Found files in Drive folder but none match the expected checkpoint "
            'naming pattern "sisyphus_ppo_{steps}_steps.zip".'
        )

    filename = os.path.basename(best_entry.path)
    local_path = os.path.join(local_dir, filename)

    # Skip download if already cached locally
    if os.path.exists(local_path):
        print(f"Using cached checkpoint: {filename}")
    else:
        print(f"Downloading {filename}...")
        gdown.download(id=best_entry.id, output=local_path, quiet=False)
        print(f"Downloaded to {local_path}")

    return local_path.replace(".zip", ""), total_steps


def find_local_checkpoint(checkpoint_dir: str) -> tuple[str, int]:
    """Find the latest checkpoint in a local directory by step count.

    Returns:
        (checkpoint_path_without_extension, total_steps)
    """
    final_path = os.path.join(checkpoint_dir, "sisyphus_ppo_final.zip")
    if os.path.exists(final_path):
        return final_path.replace(".zip", ""), SCHEDULE[-1]["end_step"]

    files = glob.glob(os.path.join(checkpoint_dir, "sisyphus_ppo_*.zip"))
    if not files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}.")

    step_re = re.compile(r"sisyphus_ppo_(\d+)_steps\.zip")
    parsed = []
    for f in files:
        match = step_re.search(os.path.basename(f))
        if match:
            parsed.append((int(match.group(1)), f))

    if not parsed:
        raise FileNotFoundError(
            f"Found .zip files but none match the expected naming pattern "
            f'in {checkpoint_dir}.'
        )

    parsed.sort(key=lambda x: x[0], reverse=True)
    latest_steps, latest_path = parsed[0]
    return latest_path.replace(".zip", ""), latest_steps


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a trained agent at its latest curriculum stage"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a specific local checkpoint (without .zip). "
        "Skips Google Drive download.",
    )
    parser.add_argument(
        "--output-dir",
        default="renders_preview",
        help="Output directory for MP4s (default: renders_preview)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Max simulation steps per episode (default: 1000)",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Output video framerate (default: 30)"
    )
    parser.add_argument(
        "--width", type=int, default=1920, help="Video width in pixels (default: 1920)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Video height in pixels (default: 1080)",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip rendering the zero-action baseline",
    )
    args = parser.parse_args()

    # --- Find checkpoint ---
    if args.checkpoint:
        checkpoint_path = args.checkpoint.replace(".zip", "")
        match = re.search(r"sisyphus_ppo_(\d+)_steps", checkpoint_path)
        if match:
            total_steps = int(match.group(1))
        elif "final" in os.path.basename(checkpoint_path):
            total_steps = SCHEDULE[-1]["end_step"]
        else:
            total_steps = SCHEDULE[-1]["end_step"]
            print(
                f"Warning: Could not parse step count from checkpoint name. "
                f"Assuming final phase ({total_steps:,} steps)."
            )
    else:
        # Check for locally cached checkpoints first, otherwise download
        if os.path.isdir(LOCAL_CHECKPOINT_DIR) and glob.glob(
            os.path.join(LOCAL_CHECKPOINT_DIR, "sisyphus_ppo_*.zip")
        ):
            checkpoint_path, total_steps = find_local_checkpoint(LOCAL_CHECKPOINT_DIR)
            print(f"Found cached checkpoint locally.")
        else:
            checkpoint_path, total_steps = download_latest_checkpoint(
                DRIVE_FOLDER_URL, LOCAL_CHECKPOINT_DIR
            )

    # --- Determine curriculum phase ---
    curriculum = CurriculumManager()
    params = curriculum.get_params(total_steps)

    print()
    print("Trained Agent Visualizer")
    print("=" * 24)
    print(f"Checkpoint:       {os.path.basename(checkpoint_path)}")
    print(f"Training steps:   {total_steps:,}")
    print(f"Curriculum phase: Phase {params.phase}")
    print(f"  Slope:          {params.slope_deg} deg")
    print(f"  Rock mass:      {params.rock_mass} kg")
    print(f"  Infinite mode:  {params.infinite}")
    print(f"  Alive bonus:    {params.alive_bonus}")
    print(f"  Upright coef:   {params.upright_coef}")
    print(f"  Forward push:   {params.forward_push_coef}")

    # --- Load model ---
    print(f"\nLoading model...")
    model = PPO.load(checkpoint_path)
    print("Model loaded.")

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Render trained policy ---
    phase_tag = f"phase{params.phase}_slope{params.slope_deg}_mass{params.rock_mass}"

    # Detect whether the checkpoint expects hand-distance observations
    # by inspecting the saved observation space dimension.
    obs_dim = model.observation_space.shape[0]
    obs_hand_dists = obs_dim > 56  # hand_dists adds 2 dims (56 → 58)

    print(f"\nRendering trained policy ({args.max_steps} steps)...")
    env = SisyphusEnv(
        slope_deg=params.slope_deg,
        rock_mass=params.rock_mass,
        infinite_mode=params.infinite,
        alive_bonus=params.alive_bonus,
        upright_coef=params.upright_coef,
        forward_push_coef=params.forward_push_coef,
        max_steps=args.max_steps,
        obs_hand_dists=obs_hand_dists,
    )
    renderer = PreviewRenderer(env.model, width=args.width, height=args.height)
    policy_path = os.path.join(args.output_dir, f"trained_policy_{phase_tag}.mp4")
    renderer.render_episode(
        env, model, policy_path,
        fps=args.fps, max_steps=args.max_steps, deterministic=True,
    )
    renderer.close()
    env.close()
    print(f"  Saved: {policy_path}")

    # --- Render zero-action baseline ---
    if not args.no_baseline:
        print(f"\nRendering zero-action baseline...")
        env_baseline = SisyphusEnv(
            slope_deg=params.slope_deg,
            rock_mass=params.rock_mass,
            infinite_mode=params.infinite,
            alive_bonus=params.alive_bonus,
            upright_coef=params.upright_coef,
            forward_push_coef=params.forward_push_coef,
            max_steps=args.max_steps,
            obs_hand_dists=obs_hand_dists,
        )
        zero_policy = ZeroPolicy(env_baseline.action_space.shape)
        renderer_baseline = PreviewRenderer(
            env_baseline.model, width=args.width, height=args.height
        )
        baseline_path = os.path.join(args.output_dir, f"baseline_{phase_tag}.mp4")
        renderer_baseline.render_episode(
            env_baseline, zero_policy, baseline_path,
            fps=args.fps, max_steps=args.max_steps, deterministic=True,
        )
        renderer_baseline.close()
        env_baseline.close()
        print(f"  Saved: {baseline_path}")

    print(f"\nDone. Videos saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
