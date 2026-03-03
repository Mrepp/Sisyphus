# Sisyphus Humanoid

A reinforcement learning experiment that trains a MuJoCo humanoid to push a rock up an incline — forever.

Inspired by the Greek myth, the agent learns locomotion and object manipulation through curriculum learning, progressing from flat ground to steep slopes with increasingly heavy rocks. A decaying posture scaffolding system helps the humanoid learn to stand before discovering its own efficient pushing gait.

## Pipeline

```
Train (PPO, 50M steps)  →  Log trajectories (.npz)  →  Preview render (MP4)  →  Blender export (JSON)
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ and a working MuJoCo installation (>= 3.1.0).

## Quick Start

### Test the environment

```python
from env.sisyphus_env import SisyphusEnv

env = SisyphusEnv(slope_deg=5.0, rock_mass=8.0, max_steps=1000)
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
```

### Train locally

```python
from train.train_ppo import create_env, create_model, train

vec_env = create_env(num_envs=16, slope=0.0, rock_mass=5.0)
model = create_model(vec_env)
model = train(model, total_timesteps=50_000_000, use_curriculum=True)
model.save("sisyphus_ppo_final")
```

### Train on Colab

Open `notebooks/sisyphus_train.ipynb` in Google Colab. Checkpoints and preview renders save to Google Drive automatically.

### Render a checkpoint

```bash
python scripts/render_checkpoint.py \
  --checkpoint logs/checkpoints/sisyphus_ppo_500000_steps \
  --output renders_preview/test.mp4 \
  --episodes 3 --slope 10.0 --mass 12.0 --max-steps 1000
```

### Replay a trajectory

```bash
python scripts/replay_trajectory.py replays/episode_0_checkpoint_5.npz
```

### Export to Blender

```python
from export.blender_export import export_for_blender

export_for_blender(
    "replays/episode_0_checkpoint_5.npz",
    "blender_export/scene",
    target_fps=60.0
)

# Outputs: joint_animation.json, rock_trajectory.json,
#          terrain_heightfield.npz, scene_metadata.json
```

Compatible with [danieldugas/blender_mujoco](https://github.com/danieldugas/blender_mujoco) for armature import.

## Curriculum

Training uses a four-phase curriculum over 50M steps:

| Phase | Slope | Rock Mass | Infinite | Posture Scaffolding |
|-------|-------|-----------|----------|---------------------|
| I     | 0°    | 5 kg      | No       | Full (alive=2.0, upright=1.0) |
| II    | 5°    | 8 kg      | No       | Decaying (alive=1.0, upright=0.3) |
| III   | 10°   | 12 kg     | No       | None |
| IV    | 15°   | 10 kg     | Yes      | None |

**Posture scaffolding** rewards standing and staying upright in early phases, then decays so the agent can develop its own efficient gait without being locked into a rigid posture.

**Infinite mode** (Phase IV) teleports the humanoid back to the start when it reaches the terrain edge, simulating an endless hill.

## Reward

```
reward = 100 * delta_rock_height
       - 0.0001 * sum(torques²)
       - 500 * fell
       + alive_bonus          (curriculum-decayed)
       + upright_bonus         (curriculum-decayed)
```

## Project Structure

```
env/                 MuJoCo environment (humanoid, rock, heightfield terrain)
train/               PPO training, curriculum schedule, callbacks
render/              Fast MP4 preview rendering from trajectories
export/              Blender-friendly JSON + heightfield export
logging_utils/       Trajectory recording (.npz)
models/              MuJoCo XML model definition
scripts/             CLI tools for rendering and replay
notebooks/           Google Colab training and evaluation notebooks
```

## Architecture

- **Humanoid:** 17 actuated joints (abdomen, legs, arms) with capsule-based collision bodies
- **Rock:** Free-floating sphere (0.15m radius), mass varies by curriculum phase
- **Terrain:** 200x50 heightfield grid, slope configured at runtime
- **PPO:** [256, 256] policy/value networks, 16 parallel envs, observation + reward normalization
- **Observation:** Joint positions/velocities, rock relative position/velocity, torso height, COM velocity

## Monitoring

TensorBoard logs are written during training:

```bash
tensorboard --logdir logs/tensorboard
```
