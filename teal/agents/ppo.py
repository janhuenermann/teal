from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

from teal.buffer import Buffer, compute_buffer_stats
from teal.agent import Agent, AgentToTrainerParams
from teal.nn.image import ImageEncoderResnet
from teal.trainer import compile, Trainer
from teal.utils.math import compute_generalised_advantage, sample_categorical


class PPOAgent(nn.Module, Agent):
    """
    Proximal Policy Optimization

    From
    "PROXIMAL POLICY OPTIMIZATION ALGORITHMS"
    https://arxiv.org/abs/1707.06347
    """

    def __init__(
        self,
        env_batch_size: int,
        lr: float = 1e-5,
        entropy_weight: float = 0.005,
        value_weight: float = 0.5,
        rollout_steps: int = 1000,
        gradient_steps: int = 100,
        gradient_batch_size: int = 512,
        max_grad_norm: float = 0.5,
        clip_eps: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.9,
    ):
        super().__init__()
        self.policy = PPOPolicy()

        self.optimizer = AdamW(self.policy.parameters(), lr=lr)
        self.entropy_weight = entropy_weight
        self.value_weight = value_weight
        self.gradient_batch_size = gradient_batch_size
        self.max_grad_norm = max_grad_norm
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.trainer_params = AgentToTrainerParams(
            inference_steps=rollout_steps, learn_steps=gradient_steps
        )
        self.buffer = Buffer(
            capacity=self.trainer_params.inference_steps * env_batch_size
        )

    @compile()
    def inference(self, observation: Dict[str, Tensor]):
        return self.policy.forward(observation)

    def add_transition(self, transition: Dict[str, np.ndarray]):
        self.buffer.insert(transition)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        return self.buffer.sample_frames(batch_size=self.gradient_batch_size)

    def on_before_learn_phase(self, trainer: Trainer):
        # First bring buffer into temporal order
        buffer_view = self.buffer.linearize()
        # Now compute advantages
        is_last = np.concatenate(
            [buffer_view["_index"][:-1, 0] != buffer_view["_index"][1:, 0], [True]]
        )
        buffer_view["advantage"] = compute_generalised_advantage(
            buffer_view["reward"],
            buffer_view["value"],
            is_last,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        # Computed normalised advantages
        adv_mu = buffer_view["advantage"].mean()
        adv_std = buffer_view["advantage"].std()
        buffer_view["normalised_advantage"] = (buffer_view["advantage"] - adv_mu) / (
            adv_std + 1e-8
        )
        trainer.metrics.log(**compute_buffer_stats(self.buffer))

    def learn_on_batch(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.optimizer.zero_grad(set_to_none=True)

        losses = self.compute_losses(batch)
        losses["loss"].backward()
        clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return losses

    @compile()
    def compute_losses(self, batch: Dict[str, Tensor]):
        action = batch["action"]
        olg_log_action_distribution = batch["log_action_distribution"]

        prediction = self.policy.forward(batch)
        log_action_distribution = prediction["log_action_distribution"]

        # Compute policy loss
        log_prob = log_action_distribution.gather(-1, action.unsqueeze(-1))
        old_log_prob = olg_log_action_distribution.gather(-1, action.unsqueeze(-1))
        ratio = (log_prob - old_log_prob).clamp(min=-10.0, max=10.0).exp().squeeze(-1)
        policy_loss = -(
            torch.min(
                batch["normalised_advantage"] * ratio,
                batch["normalised_advantage"]
                * ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps),
            )
        ).mean()

        # Compute value loss
        value_loss = F.mse_loss(
            prediction["value"], batch["value"] + batch["advantage"]
        )

        # Compute entropy loss
        entropy_loss = (
            -(log_action_distribution * log_action_distribution.exp())
            .sum(dim=-1)
            .mean()
        )

        # Track KL divergence between old and new action distributions
        kl_div = (
            (
                olg_log_action_distribution.exp()
                * (olg_log_action_distribution - log_action_distribution.detach())
            )
            .sum(dim=-1)
            .mean()
        )

        # Compute total loss
        loss = (
            policy_loss
            + self.value_weight * value_loss
            - self.entropy_weight * entropy_loss
        )

        return {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "kl_div": kl_div,
        }


class PPOPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ImageEncoderResnet(in_channels=3, in_width=64)
        self.value_head = nn.Sequential(
            nn.Linear(self.encoder.out_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.action_head = nn.Sequential(
            nn.Linear(self.encoder.out_features, 256),
            nn.ReLU(),
            nn.Linear(256, 15),
        )

    def forward(self, observation: Dict[str, Tensor]):
        image = observation["rgb"].permute(0, 3, 1, 2).float() / 255.0
        feat = self.encoder.forward(image)
        action_logits = self.action_head(feat)
        value = self.value_head(feat).squeeze(-1)
        log_action_distribution = torch.log_softmax(action_logits, dim=-1)
        action = sample_categorical(log_action_distribution.exp())
        return {
            "action": action,
            "value": value,
            "log_action_distribution": log_action_distribution,
        }
