"""Visualize a trained agent across all curriculum phases it has completed.

Downloads the latest PPO checkpoint from a Google Drive shared folder,
renders the trained policy at each curriculum phase (plus zero-action
baselines), and exports Blender-ready data for each phase.

Outputs are organized in a git-commit-tagged folder under render_output/.

Usage:
    python scripts/visualize_trained_agent.py
    python scripts/visualize_trained_agent.py --no-baseline --max-steps 2000
    python scripts/visualize_trained_agent.py --checkpoint /local/path/to/sisyphus_ppo_5000000_steps
"""

import argparse
import datetime
import glob
import json
import os
import re
import subprocess
import sys

# Ensure project root is on the path when running as a script.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from stable_baselines3 import PPO

from env.sisyphus_env import SisyphusEnv
from export.blender_export import export_for_blender
from render.preview_renderer import PreviewRenderer
from train.curriculum import SCHEDULE, CurriculumManager


DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1bq_jR1oP4aUvxDi1A8gMRFvlgNIQKEaZ"
LOCAL_CHECKPOINT_DIR = os.path.join(_PROJECT_ROOT, "checkpoints")


class ZeroPolicy:
    """Mimics the SB3 model.predict() interface but returns zero actions."""

    def __init__(self, action_shape):
        self._zeros = np.zeros(action_shape)

    def predict(self, obs, deterministic=True):
        return self._zeros, None


def _get_git_info() -> tuple[str, str]:
    """Return (short_hash, full_hash) from the project repo."""
    try:
        full = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_PROJECT_ROOT, text=True,
        ).strip()
        short = full[:7]
        return short, full
    except Exception:
        return "unknown", "unknown"


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


def _phases_up_to(total_steps: int) -> list[dict]:
    """Return all curriculum phases the checkpoint has reached, in order.

    Includes completed phases and the current (possibly in-progress) phase.
    """
    phases = []
    for entry in SCHEDULE:
        phases.append(entry)
        if total_steps < entry["end_step"]:
            break  # this is the current phase — include it and stop
    return phases


def _render_phase(
    phase_entry: dict,
    model,
    obs_hand_dists: bool,
    phase_dir: str,
    args,
    git_short: str,
    total_steps: int,
):
    """Render trained policy + optional baseline for one curriculum phase."""
    curriculum = CurriculumManager()
    params = curriculum._params_from_entry(phase_entry)

    print(f"\n--- Phase {params.phase}: slope={params.slope_deg}°, mass={params.rock_mass}kg ---")

    # --- Trained policy ---
    print(f"  Rendering trained policy ({args.max_steps} steps)...")
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

    mp4_path = os.path.join(phase_dir, "trained_policy.mp4")
    traj_path = os.path.join(phase_dir, "trajectory.npz")

    traj_meta = {
        "slope": params.slope_deg,
        "mass": params.rock_mass,
        "phase": params.phase,
        "total_steps": total_steps,
        "git_commit": git_short,
    }

    renderer.render_episode(
        env, model, mp4_path,
        fps=args.fps, max_steps=args.max_steps, deterministic=True,
        trajectory_path=traj_path, trajectory_metadata=traj_meta,
    )
    renderer.close()
    env.close()
    print(f"    Saved: {mp4_path}")

    # --- Blender export ---
    blender_dir = os.path.join(phase_dir, "blender")
    print(f"  Exporting Blender data...")
    export_for_blender(traj_path, blender_dir, target_fps=60.0)
    print(f"    Saved: {blender_dir}/")

    # --- Baseline ---
    if not args.no_baseline:
        print(f"  Rendering zero-action baseline...")
        env_bl = SisyphusEnv(
            slope_deg=params.slope_deg,
            rock_mass=params.rock_mass,
            infinite_mode=params.infinite,
            alive_bonus=params.alive_bonus,
            upright_coef=params.upright_coef,
            forward_push_coef=params.forward_push_coef,
            max_steps=args.max_steps,
            obs_hand_dists=obs_hand_dists,
        )
        zero_policy = ZeroPolicy(env_bl.action_space.shape)
        renderer_bl = PreviewRenderer(env_bl.model, width=args.width, height=args.height)
        baseline_path = os.path.join(phase_dir, "baseline.mp4")
        renderer_bl.render_episode(
            env_bl, zero_policy, baseline_path,
            fps=args.fps, max_steps=args.max_steps, deterministic=True,
        )
        renderer_bl.close()
        env_bl.close()
        print(f"    Saved: {baseline_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a trained agent across all curriculum phases"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a specific local checkpoint (without .zip). "
        "Skips Google Drive download.",
    )
    parser.add_argument(
        "--output-dir",
        default="render_output",
        help="Base output directory (default: render_output)",
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

    # --- Git info ---
    git_short, git_full = _get_git_info()

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
        # Always check Google Drive for the latest checkpoint
        try:
            checkpoint_path, total_steps = download_latest_checkpoint(
                DRIVE_FOLDER_URL, LOCAL_CHECKPOINT_DIR
            )
        except Exception as e:
            print(f"Could not access Google Drive ({e}), checking local cache...")
            checkpoint_path, total_steps = find_local_checkpoint(LOCAL_CHECKPOINT_DIR)

    # --- Create output folder ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{git_short}_{total_steps}_{timestamp}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # --- Determine phases to render ---
    phases = _phases_up_to(total_steps)
    current_phase = CurriculumManager().get_params(total_steps)

    print()
    print("Trained Agent Visualizer")
    print("=" * 40)
    print(f"Git commit:       {git_short}")
    print(f"Checkpoint:       {os.path.basename(checkpoint_path)}")
    print(f"Training steps:   {total_steps:,}")
    print(f"Current phase:    Phase {current_phase.phase}")
    print(f"Phases to render: {len(phases)} (I through {phases[-1]['phase']})")
    print(f"Output:           {run_dir}/")

    # --- Load model ---
    print(f"\nLoading model...")
    model = PPO.load(checkpoint_path)
    print("Model loaded.")

    # Detect whether the checkpoint expects hand-distance observations
    obs_dim = model.observation_space.shape[0]
    obs_hand_dists = obs_dim > 56  # hand_dists adds 2 dims (56 -> 58)

    # --- Render each phase ---
    manifest_phases = []
    for phase_entry in phases:
        phase_tag = f"phase{phase_entry['phase']}_slope{phase_entry['slope']}_mass{phase_entry['mass']}"
        phase_dir = os.path.join(run_dir, phase_tag)
        os.makedirs(phase_dir, exist_ok=True)

        _render_phase(
            phase_entry, model, obs_hand_dists, phase_dir, args,
            git_short, total_steps,
        )

        manifest_phases.append({
            "phase": phase_entry["phase"],
            "slope_deg": phase_entry["slope"],
            "rock_mass_kg": phase_entry["mass"],
            "infinite_mode": phase_entry["infinite"],
            "directory": phase_tag,
        })

    # --- Write manifest ---
    manifest = {
        "git_commit_short": git_short,
        "git_commit_full": git_full,
        "checkpoint": os.path.basename(checkpoint_path),
        "total_training_steps": total_steps,
        "timestamp": timestamp,
        "max_steps_per_episode": args.max_steps,
        "fps": args.fps,
        "resolution": f"{args.width}x{args.height}",
        "baseline_included": not args.no_baseline,
        "phases": manifest_phases,
    }
    manifest_path = os.path.join(run_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. All outputs saved to {run_dir}/")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
