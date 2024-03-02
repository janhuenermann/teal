from pathlib import Path
from typing import Optional, Union
import numpy as np
import torch
from torch import nn
from tqdm import trange

from teal.env import Env
from teal.utils.video import VideoWriter


def evaluate_policy(
    policy: nn.Module,
    env: Env,
    output_video_path: Optional[Union[str, Path]] = None,
    verbose=False,
) -> np.ndarray:
    """
    Runs evaluation of the policy and returns an array of episode returns.
    """
    env_batch_size = next(iter(env.observe().values())).shape[0]
    device = next(policy.parameters()).device  # type: ignore

    # outputs to collect
    mask = np.ones(env_batch_size, dtype=bool)
    cum_reward = np.zeros(env_batch_size, dtype=np.float32)
    collected_rewards = np.array([])
    writer = VideoWriter(output_video_path, fps=15) if output_video_path else None

    pbar = trange(10_000, ncols=120, disable=not verbose)
    for _ in pbar:
        obs = env.observe()

        th_obs = {k: torch.from_numpy(v).to(device) for k, v in obs.items()}
        prediction = policy(th_obs)
        th_action = prediction["action"]

        if writer is not None:  # before cpu() to allow async policy inference
            writer.write_frame(obs["rgb"][0])

        action = th_action.cpu().numpy()
        reward, is_done = env.act(action)

        # update metrics
        cum_reward += reward
        collected_rewards = np.append(collected_rewards, cum_reward[is_done])
        cum_reward[is_done] = 0.0

        # run evaluation until each env has finished at least one episode
        mask = mask & (~is_done)
        if not np.any(mask):
            break

        pbar.set_postfix(dict(n_running=mask.sum(), n_collected=len(collected_rewards)))

    if writer is not None:
        writer.close()

    return collected_rewards
