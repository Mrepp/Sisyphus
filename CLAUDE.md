# CLAUDE.md

## Project Overview

Sisyphus Humanoid — RL experiment training a MuJoCo humanoid to push a rock uphill using PPO with curriculum learning.

## Tech Stack

- MuJoCo (physics), Gymnasium (env interface), Stable-Baselines3 (PPO)
- NumPy, ImageIO (video), TensorBoard (monitoring)
- Blender export pipeline (JSON + heightfield)

## Key Directories

- `env/` — Core MuJoCo environment (`SisyphusEnv`)
- `train/` — PPO training (`train_ppo.py`), curriculum (`curriculum.py`), callbacks
- `render/` — MP4 preview renderer
- `export/` — Blender export (joint animation JSON, rock trajectory, terrain heightfield)
- `logging_utils/` — Trajectory logger (.npz format)
- `models/` — MuJoCo XML model (`sisyphus_humanoid.xml`)
- `scripts/` — CLI tools: `render_checkpoint.py`, `replay_trajectory.py`
- `notebooks/` — Colab notebooks for training and evaluation

## Setup & Data Flow

- **Code**: Pull from GitHub — `https://github.com/Mrepp/Sisyphus.git`
- **Checkpoints / heavy artifacts**: Stored on Google Drive at `drive/MyDrive/sisyphus/`
  - `checkpoints/` — SB3 `.zip` files + `vec_normalize_*.pkl`
  - `replays/` — Trajectory `.npz` files
  - `renders_preview/` — Preview MP4s
  - `tensorboard/` — TensorBoard logs
  - `blender_export/` — Blender-ready JSON + heightfield exports

Colab notebooks (`notebooks/`) are the primary training interface. They clone the repo from GitHub and read/write checkpoints to Google Drive so artifacts persist across sessions.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Render a checkpoint from Drive
python scripts/render_checkpoint.py \
    --checkpoint /content/drive/MyDrive/sisyphus/checkpoints/sisyphus_ppo_final \
    --output renders_preview/eval.mp4

# Replay a trajectory interactively
python scripts/replay_trajectory.py replays/episode_0_checkpoint_5.npz

# TensorBoard (local or Colab)
tensorboard --logdir logs/tensorboard            # local
tensorboard --logdir /content/drive/MyDrive/sisyphus/tensorboard  # Colab
```

## Architecture Notes

- Environment uses a 200x50 heightfield grid with runtime-configurable slope
- Humanoid has 17 actuated joints; observation includes joint state, rock position, torso height, COM velocity
- Curriculum has 4 phases (0°→15° slope, 20→50 kg rock mass) with decaying posture scaffolding
- Phase IV enables "infinite mode" — teleport to start when reaching terrain edge
- Training uses 16 parallel SubprocVecEnv with VecNormalize
- Trajectories are logged as compressed .npz files with qpos, qvel, rock_pos, torques, rewards, contacts
- Preview renders are 1920x1080 MP4 at 30 FPS; Blender export targets 60 FPS

## Conventions

- Training outputs go to `logs/` (gitignored)
- Trajectory replays go to `replays/` (gitignored)
- Rendered videos go to `renders_preview/` or `renders_cinematic/` (gitignored)
- Blender project files go to `blender_project/` (gitignored)
- All heavy/generated artifacts are gitignored; only source code and the MuJoCo XML are tracked
