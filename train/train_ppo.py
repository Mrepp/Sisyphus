"""PPO training functions for Sisyphus — designed to be called from Colab notebooks.

Usage from notebook:
    from train.train_ppo import create_env, create_model, train
    vec_env = create_env(num_envs=16, slope=0.0, rock_mass=5.0)
    model = create_model(vec_env)
    train(model, total_timesteps=5_000_000, ...)
"""

from __future__ import annotations

import os
import logging
import multiprocessing
import platform
from typing import TYPE_CHECKING

from env.sisyphus_env import SisyphusEnv
from train.curriculum import CurriculumManager

if TYPE_CHECKING:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

logger = logging.getLogger(__name__)


def get_hardware_config(
    num_envs: int | None = None,
    profile: str = "auto",
) -> dict:
    """Return recommended training hyperparameters for the detected hardware.

    Args:
        num_envs: Override number of parallel environments. If None, auto-detect.
        profile: One of "colab_a100", "dedicated_a100", "cpu", or "auto".

    Returns:
        Dict with keys: num_envs, n_steps, batch_size, device, net_arch, profile.
    """
    import torch  # lazy import to avoid subprocess C++ conflicts

    cpu_count = multiprocessing.cpu_count()
    has_cuda = torch.cuda.is_available()

    if profile == "auto":
        if has_cuda:
            if cpu_count <= 12:
                profile = "colab_a100"
            else:
                profile = "dedicated_a100"
        else:
            profile = "cpu"

    configs = {
        "cpu": {
            "num_envs": min(cpu_count, 16),
            "n_steps": 2048,
            "batch_size": 1024,
            "device": "cpu",
            "net_arch": [256, 256],
        },
        "colab_a100": {
            "num_envs": 10,
            "n_steps": 4096,
            "batch_size": 2048,
            "device": "cuda",
            "net_arch": [512, 512],
        },
        "dedicated_a100": {
            "num_envs": min(cpu_count - 4, 64),
            "n_steps": 2048,
            "batch_size": 4096,
            "device": "cuda",
            "net_arch": [512, 512],
        },
    }

    config = configs[profile]
    if num_envs is not None:
        config["num_envs"] = num_envs
    config["profile"] = profile

    return config


def setup_torch_optimizations():
    """Configure PyTorch for optimal A100 performance."""
    import torch  # lazy import to avoid subprocess C++ conflicts

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"TF32 matmul enabled: {torch.backends.cuda.matmul.allow_tf32}")


def make_env(slope: float, rock_mass: float, rank: int, max_steps: int = 1000):
    """Factory for creating a single env instance (used by SubprocVecEnv)."""
    def _init():
        from stable_baselines3.common.monitor import Monitor

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
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    # macOS: "forkserver" avoids C++ static-init crashes (MuJoCo + PyTorch).
    # Linux/Colab: "spawn" gives each subprocess a clean CUDA context,
    # preventing cuBLAS factory double-registration.
    if platform.system() == "Darwin":
        start_method = "forkserver"
    else:
        start_method = "spawn"

    env_fns = [make_env(slope, rock_mass, i, max_steps) for i in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns, start_method=start_method)
    vec_env = VecNormalize(vec_env, norm_obs=norm_obs, norm_reward=norm_reward)
    return vec_env


def create_model(
    vec_env: VecNormalize,
    tensorboard_log: str = "logs/tensorboard",
    hardware_config: dict | None = None,
    **kwargs,
) -> PPO:
    """Create a PPO model with Sisyphus-tuned defaults.

    Args:
        vec_env: Vectorized environment.
        tensorboard_log: TensorBoard log directory.
        hardware_config: Dict from get_hardware_config() to override n_steps,
            batch_size, device, and net_arch. Any explicit kwarg still wins.
    """
    from stable_baselines3 import PPO

    hw = hardware_config or {}
    net_arch = hw.get("net_arch", [256, 256])

    defaults = dict(
        learning_rate=3e-4,
        n_steps=hw.get("n_steps", 2048),
        batch_size=hw.get("batch_size", 1024),
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        target_kl=0.03,
        ent_coef=0.01,
        verbose=1,
        device=hw.get("device", "auto"),
        tensorboard_log=tensorboard_log,
        policy_kwargs=dict(
            net_arch=[dict(pi=net_arch, vf=net_arch)],
            log_std_init=-1.0,  # Initial std ≈ 0.37 (reduces flailing)
        ),
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
    render_enabled: bool = True,
    reset_num_timesteps: bool = True,
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
        render_enabled: Render preview MP4s at each checkpoint. Disable on Colab.
        reset_num_timesteps: Reset step counter (False for chunked training).
    """
    from stable_baselines3.common.callbacks import CheckpointCallback
    from train.callbacks import CurriculumCallback, TrajectoryRenderCallback

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

    if eval_env is None:
        eval_env = SisyphusEnv(
            slope_deg=0.0, rock_mass=20.0, max_steps=1000,
        )

    # Wrap eval env for normalized observations (must match training)
    from stable_baselines3.common.vec_env import (
        DummyVecEnv, VecNormalize as VecNorm,
    )
    eval_vec_env = VecNorm(
        DummyVecEnv([lambda: eval_env]),
        norm_obs=True,
        norm_reward=False,
        training=False,
    )

    if use_curriculum:
        curriculum = curriculum or CurriculumManager()
        callbacks.append(
            CurriculumCallback(
                curriculum,
                eval_env=eval_env,
                verbose=1,
            )
        )

    callbacks.append(
        TrajectoryRenderCallback(
            eval_env=eval_env,
            eval_vec_env=eval_vec_env,
            training_env=model.env,
            save_freq=save_freq,
            replay_dir=replay_dir,
            render_dir=render_dir,
            render_enabled=render_enabled,
            verbose=1,
        )
    )

    logger.info(f"Starting training: {total_timesteps} steps, {num_envs} envs")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        reset_num_timesteps=reset_num_timesteps,
    )
    logger.info("Training complete.")

    return model


def load_model(path: str, vec_env: VecNormalize | None = None) -> PPO:
    """Load a saved PPO checkpoint."""
    from stable_baselines3 import PPO

    return PPO.load(path, env=vec_env)


def save_vec_normalize(vec_env: VecNormalize, path: str):
    """Save VecNormalize statistics for later loading."""
    vec_env.save(path)


def load_vec_normalize(vec_env: SubprocVecEnv, path: str) -> VecNormalize:
    """Load VecNormalize statistics onto a vectorized env."""
    from stable_baselines3.common.vec_env import VecNormalize

    return VecNormalize.load(path, vec_env)
