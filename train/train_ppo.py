"""PPO training functions for Sisyphus — designed to be called from Colab notebooks.

Usage from notebook:
    from train.train_ppo import create_env, create_model, train
    vec_env = create_env(num_envs=16, slope=0.0, rock_mass=5.0)
    model = create_model(vec_env)
    train(model, total_timesteps=5_000_000, ...)
"""

import os
import logging

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from env.sisyphus_env import SisyphusEnv
from train.curriculum import CurriculumManager
from train.callbacks import CurriculumCallback, TrajectoryRenderCallback

logger = logging.getLogger(__name__)


def make_env(slope: float, rock_mass: float, rank: int, max_steps: int = 1000):
    """Factory for creating a single env instance (used by SubprocVecEnv)."""
    def _init():
        env = SisyphusEnv(slope_deg=slope, rock_mass=rock_mass, max_steps=max_steps)
        env = Monitor(env)
        return env
    return _init


def create_env(
    num_envs: int = 16,
    slope: float = 0.0,
    rock_mass: float = 20.0,
    max_steps: int = 1000,
    norm_obs: bool = True,
    norm_reward: bool = True,
) -> VecNormalize:
    """Create a vectorized, normalized environment for SB3.

    Returns:
        VecNormalize wrapping SubprocVecEnv.
    """
    env_fns = [make_env(slope, rock_mass, i, max_steps) for i in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecNormalize(vec_env, norm_obs=norm_obs, norm_reward=norm_reward)
    return vec_env


def create_model(
    vec_env: VecNormalize,
    tensorboard_log: str = "logs/tensorboard",
    **kwargs,
) -> PPO:
    """Create a PPO model with Sisyphus-tuned defaults.

    Any kwarg overrides the defaults below.
    """
    defaults = dict(
        learning_rate=lambda f: 3e-4 * f,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        target_kl=0.03,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=tensorboard_log,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
    )
    defaults.update(kwargs)
    return PPO("MlpPolicy", vec_env, **defaults)


def train(
    model: PPO,
    total_timesteps: int = 50_000_000,
    checkpoint_dir: str = "logs/checkpoints",
    replay_dir: str = "replays",
    render_dir: str = "renders_preview",
    checkpoint_freq: int = 500_000,
    use_curriculum: bool = True,
    curriculum: CurriculumManager | None = None,
    eval_env: SisyphusEnv | None = None,
):
    """Run the PPO training loop.

    Args:
        model: PPO model (from create_model).
        total_timesteps: Total training steps.
        checkpoint_dir: Where to save SB3 checkpoints.
        replay_dir: Where to save trajectory .npz files.
        render_dir: Where to save preview MP4s.
        checkpoint_freq: Steps between checkpoints.
        use_curriculum: Enable curriculum transitions.
        curriculum: CurriculumManager instance (created if None).
        eval_env: Unwrapped SisyphusEnv for evaluation (created if None).
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(replay_dir, exist_ok=True)
    os.makedirs(render_dir, exist_ok=True)

    num_envs = model.env.num_envs
    save_freq = max(checkpoint_freq // num_envs, 1)

    callbacks = [
        CheckpointCallback(
            save_freq=save_freq,
            save_path=checkpoint_dir,
            name_prefix="sisyphus_ppo",
        ),
    ]

    if use_curriculum:
        curriculum = curriculum or CurriculumManager()
        callbacks.append(CurriculumCallback(curriculum, verbose=1))

    if eval_env is None:
        eval_env = SisyphusEnv(slope_deg=0.0, rock_mass=20.0, max_steps=1000)

    callbacks.append(
        TrajectoryRenderCallback(
            eval_env=eval_env,
            save_freq=checkpoint_freq,
            replay_dir=replay_dir,
            render_dir=render_dir,
            verbose=1,
        )
    )

    logger.info(f"Starting training: {total_timesteps} steps, {num_envs} envs")
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    logger.info("Training complete.")

    return model


def load_model(path: str, vec_env: VecNormalize | None = None) -> PPO:
    """Load a saved PPO checkpoint."""
    return PPO.load(path, env=vec_env)


def save_vec_normalize(vec_env: VecNormalize, path: str):
    """Save VecNormalize statistics for later loading."""
    vec_env.save(path)


def load_vec_normalize(vec_env: SubprocVecEnv, path: str) -> VecNormalize:
    """Load VecNormalize statistics onto a vectorized env."""
    return VecNormalize.load(path, vec_env)
