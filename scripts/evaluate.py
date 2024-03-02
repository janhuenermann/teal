from pathlib import Path
from hydra_zen import zen, store
import numpy as np

import torch

from teal.env import Env  # pylint: disable=unused-import
from teal.trainer import find_system_device
from teal.utils.evaluation import evaluate_policy


def evaluate(model_path: Path, env: Env):
    policy = torch.jit.load(model_path, map_location="cpu")  # type: ignore

    device = find_system_device()
    policy.to(device)  # type: ignore

    episode_returns = evaluate_policy(
        policy=policy, env=env, output_video_path="video.mp4", verbose=True
    )

    avg_reward = episode_returns.mean()
    median_reward = np.median(episode_returns)
    std_reward = episode_returns.std()
    print(
        f"Average reward: {avg_reward:.4f} +- {std_reward:.4f} over {len(episode_returns)} episodes"
    )
    print(f"Median reward: {median_reward:.4f}")
    print(f"20th quantile: {np.quantile(episode_returns, 0.2):.4f}")
    print(f"80th quantile: {np.quantile(episode_returns, 0.8):.4f}")


store(evaluate, name="evaluate")


if __name__ == "__main__":
    store.add_to_hydra_store()
    zen(evaluate).hydra_main(
        config_name="evaluate",
        config_path=None,
        version_base=None,
    )
