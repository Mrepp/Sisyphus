"""Visualize a trained agent across all curriculum phases it has completed.

Loads a PPO checkpoint, renders the trained policy at each curriculum phase
(plus zero-action baselines), and exports Blender-ready data for each phase.

By default, connects to Google Drive, downloads the latest checkpoint, and
renders it. Use --steps to pick a specific checkpoint, or --checkpoint for
a local file.

Outputs are organized in a git-commit-tagged folder under render_output/.

Usage:
    python scripts/visualize_trained_agent.py                    # latest from Drive
    python scripts/visualize_trained_agent.py --steps 52331648   # specific step count
    python scripts/visualize_trained_agent.py --checkpoint /local/path/to/model
    python scripts/visualize_trained_agent.py --no-baseline --max-steps 2000
"""

import argparse
import datetime
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


def _phases_up_to(total_steps: int) -> list[dict]:
    """Return all curriculum phases to render, in order.

    With metric-gated promotion we can't determine the exact phase from
    step count alone, so we return all phases — harmless to render extra ones.
    """
    return list(SCHEDULE)


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
        "--steps",
        type=int,
        default=None,
        help="Training step count of a specific checkpoint to load. "
             "Without this, the latest checkpoint is used.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a local model .zip file (e.g. checkpoints/sisyphus_ppo_52331648_steps.zip).",
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
    parser.add_argument(
        "--drive-folder-id",
        default=None,
        help="Google Drive folder ID for the checkpoints folder. "
             "Cached after first use.",
    )
    args = parser.parse_args()

    # --- Git info ---
    git_short, git_full = _get_git_info()

    # --- Find checkpoint ---
    if args.checkpoint:
        # Explicit local path
        checkpoint_path = args.checkpoint.replace(".zip", "")
        match = re.search(r"sisyphus_ppo_(\d+)_steps", checkpoint_path)
        if match:
            total_steps = int(match.group(1))
        elif "final" in os.path.basename(checkpoint_path):
            total_steps = 100_000_000
        else:
            total_steps = 100_000_000
            print(
                f"Warning: Could not parse step count from checkpoint name. "
                f"Assuming final phase ({total_steps:,} steps)."
            )
    elif args.steps is not None:
        # Specific step count — check local, then Drive
        total_steps = args.steps
        filename = f"sisyphus_ppo_{total_steps}_steps.zip"
        local_path = os.path.join(LOCAL_CHECKPOINT_DIR, filename)

        if not os.path.exists(local_path):
            from scripts.drive_utils import download_checkpoint
            try:
                local_path = download_checkpoint(
                    filename, LOCAL_CHECKPOINT_DIR,
                    args.drive_folder_id,
                )
            except Exception as e:
                print(f"Error: {e}")
                sys.exit(1)

        checkpoint_path = local_path.replace(".zip", "")
        print(f"Using checkpoint: {os.path.basename(local_path)}")
    else:
        # No args — get the latest checkpoint from Drive
        from scripts.drive_utils import download_latest_checkpoint
        try:
            local_path, total_steps = download_latest_checkpoint(
                LOCAL_CHECKPOINT_DIR, args.drive_folder_id,
            )
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

        if total_steps == 0:
            total_steps = 100_000_000

        checkpoint_path = local_path.replace(".zip", "")
        print(f"Using checkpoint: {os.path.basename(local_path)}")

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
    # Base obs = 60 dims (qpos + qvel + rock_rel + rock_vel + torso_h + com_vel
    #   + foot contacts + agent_touching + rock_y).
    # With obs_hand_dists=True: +2 dims → 62 total.
    obs_hand_dists = obs_dim > 60

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
