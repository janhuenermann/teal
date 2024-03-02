import copy
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from torch import Tensor, nn
import torch
from torch.optim import AdamW
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from teal.buffer import Buffer, compute_buffer_stats
from teal.agent import Agent, AgentToTrainerParams
from teal.nn.image import ImageEncoderResnet, ImageDecoderResnet
from teal.nn.mlp import MLP
from teal.trainer import (
    cast_dict_to_numpy,
    cast_dict_to_system_device,
    compile,
    Trainer,
)
import teal.utils.distributions as distr
from teal.utils.visualisation import draw_text, video_bar_chart


class DreamerAgent(nn.Module, Agent):
    """
    DreamerV3.

    From
    "MASTERING DIVERSE DOMAINS THROUGH WORLD MODELS"
    https://arxiv.org/abs/2301.04104
    """

    def __init__(
        self,
        model_lr: float = 1e-4,
        actor_lr: float = 4e-5,
        batch_size: int = 32,
        max_grad_norm: float = 0.5,
    ):
        super().__init__()
        self.policy: DreamerPolicy = DreamerPolicy()

        self.wm_optimizer = AdamW(self.policy.world_model.parameters(), lr=model_lr)
        self.actor_critic_optimizer = AdamW(
            list(self.policy.actor.parameters())
            + list(self.policy.critic.parameters()),
            lr=actor_lr,
            eps=1e-5,
        )

        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        self.trainer_params = AgentToTrainerParams(
            inference_steps=1000, learn_steps=200
        )
        self.buffer = Buffer(capacity=500_000)

    @compile()
    def inference(self, observation: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.policy.forward(observation)

    def add_transition(self, transition: Dict[str, np.ndarray]):
        self.buffer.insert(transition)

    def on_before_learn_phase(self, trainer: Trainer):
        trainer.metrics.log(**compute_buffer_stats(self.buffer))

        if trainer.global_step % 25 == 0:
            logging.info("visualising dreamer.... 💭")
            video = self.visualise()
            trainer.logger.log_video("imagination", video, fps=5)
            logging.info("visualisation done!")

    def sample_batch(self) -> Dict[str, np.ndarray]:
        return self.buffer.sample_sequences(self.batch_size, sequence_size=50)

    def learn_on_batch(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # ---- World model update ----

        self.wm_optimizer.zero_grad(set_to_none=True)

        losses_wm, h_z = self.policy.world_model.compute_loss(batch)

        losses_wm["loss_wm"].backward()
        clip_grad_norm_(self.policy.world_model.parameters(), 100.0)
        self.wm_optimizer.step()

        # ---- Actor and critic update ----

        self.actor_critic_optimizer.zero_grad(set_to_none=True)

        # Imagination starting from every state in the sequence
        losses_a_c = self.policy.compute_actor_critic_losses(
            h_z=h_z.detach().flatten(0, 1),
            initial_done=batch["is_done"].flatten(0, 1),
        )

        # fine to backprop through both actor and critic loss simultaneously,
        # since the actor and critic are independent.
        losses_a_c["loss_a_c"].backward()

        clip_grad_norm_(self.policy.actor.parameters(), 10.0)
        clip_grad_norm_(self.policy.critic.parameters(), 10.0)
        self.actor_critic_optimizer.step()

        self.policy.critic.update_slow_critic()

        return {**losses_wm, **losses_a_c}

    def visualise(self):
        sequences = cast_dict_to_system_device(
            self.buffer.sample_sequences(batch_size=8, sequence_size=50)
        )

        th_data = self.policy.imagine_videos(sequences)
        data = cast_dict_to_numpy(th_data)
        keys = ["GT", "OBS_ACT", "OBS0_ACT", "OBS0", "NONE"]

        for key in keys:
            reward_key = f"{key}_REWARD"
            if reward_key in data:
                reward = data[reward_key]  # shape [T, B]
                reward = np.pad(reward, ((0, 0), (0, 1)), constant_values=0.0)
                batch_size = reward.shape[0]
                video_width = data[key].shape[2] // batch_size
                bar_charts = np.concatenate(
                    [
                        video_bar_chart(reward[i], width=video_width, height=21)
                        for i in range(batch_size)
                    ],
                    axis=2,
                )
                data[key] = np.concatenate([data[key], bar_charts], axis=1)

        # video is [T, H, W, C] uint8
        video = np.concatenate([data[key] for key in keys], axis=1)
        # add a section_height x label_width box on the left
        # with the key name in white font
        label_width = 128
        section_offsets = np.roll(np.cumsum([data[key].shape[1] for key in keys]), 1)
        section_offsets[0] = 0
        border = np.zeros((video.shape[1], label_width, 3), dtype=np.uint8)
        for key, section_offset in zip(keys, section_offsets):
            section_size = data[key].shape[1]
            center = (label_width // 2, section_offset + section_size // 2)
            draw_text(border, key, center, fontsize=0.4)
        border = np.broadcast_to(border, (video.shape[0], *border.shape))
        video = np.concatenate([border, video], axis=2)
        return video


class DreamerPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.world_model = WorldModel()
        self.actor = Actor(in_features=self.world_model.h_z_dim)
        self.critic = Critic(in_features=self.world_model.h_z_dim)

        # Set all biases to zero
        for key, value in self.named_parameters():
            if key.endswith("bias"):
                nn.init.zeros_(value)

    def forward(self, obs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        h, z_post = self.world_model.forward(obs)
        z = z_post.sample()
        action_distr = self.actor.forward(torch.cat([h, z], dim=-1))
        action_one_hot = action_distr.sample()
        h = self.world_model.dynamics_step(h=h, z=z, a=action_one_hot)
        action_index = action_one_hot.argmax(-1)
        return {"action": action_index, "state_h": h}

    @compile()
    def compute_actor_critic_losses(
        self,
        h_z: Tensor,
        initial_done: Tensor,
        rollout_length: int = 15,
    ):
        # ---- Imagination ----

        # with torch.no_grad():
        imag_h_z, imag_a_one_hot = self.world_model.imagine_sequence(
            time_steps=rollout_length,
            actor=self.actor,
            initial_h_z=h_z,
        )

        imag_returns, imag_normed_adv = self.critic.compute_returns(
            h_z=imag_h_z, world_model=self.world_model
        )

        cont = distr.mode(self.world_model.decode_cont(imag_h_z))
        cont = torch.cat([~initial_done[:, None], cont], dim=1)
        # TODO: maybe weight by discount factor too?
        weight = torch.cumprod(cont.float(), dim=1)

        # TODO: move to torch.no_grad() once torch 2.1.0 is released
        weight = weight.detach()
        cont = cont.detach()
        imag_returns = imag_returns.detach()
        imag_normed_adv = imag_normed_adv.detach()
        imag_h_z = imag_h_z.detach()
        imag_a_one_hot = imag_a_one_hot.detach()

        imag_a_distr = self.actor.forward(imag_h_z[:, :-1])
        imag_val = self.critic.forward(imag_h_z[:, :-1])

        # ---- Actor losses ----

        ent = imag_a_distr.entropy()
        loss_reinforce = -imag_normed_adv * distr.log_prob(imag_a_distr, imag_a_one_hot)

        loss_actor = loss_reinforce - 0.005 * ent
        loss_actor = torch.mean(loss_actor * weight[:, :-1])

        # ---- Critic losses ----

        # TODO: log prob regularisation of critic
        loss_critic = -imag_val.log_prob(imag_returns)
        loss_critic = (loss_critic * weight[:, :-1]).mean()

        # ---- Join losses ----
        loss_a_c = loss_actor + loss_critic

        return {
            "loss_a_c": loss_a_c,
            "loss_actor": loss_actor,
            "loss_critic": loss_critic,
            "loss_reinforce": loss_reinforce.detach().mean(),
            "policy_ent": ent.detach().mean(),
            "critic_scale": self.critic.value_scale(),
            "imag_returns": imag_returns.detach().mean(),
        }

    @torch.inference_mode()
    def imagine_videos(self, batch: Dict[str, Tensor]):
        image = batch["rgb"].permute(0, 1, 4, 2, 3).float() / 255.0
        action = F.one_hot(batch["action"], num_classes=15).float()

        obs_feat = self.world_model.encoder.forward(image)
        output: Dict[str, Tensor] = {}

        # [B, T, H, W, C] -> [T, H, B * W, C]
        output["GT"] = cast_to_video_grid(image)
        output["GT_REWARD"] = batch["reward"][:, :-1]

        # ---- Observation + action conditioning ----

        h_z, _ = self.world_model.observe_sequence(obs_feat, action[:, :-1])
        output["OBS_ACT"] = cast_to_video_grid(self.world_model.decode_image(h_z))
        output["OBS_ACT_REWARD"] = self.world_model.decode_reward(h_z).mode()

        # ---- First observation + action conditioning ----

        h_z, _ = self.world_model.imagine_sequence(
            time_steps=action.shape[1],
            actor=self.actor,
            o=obs_feat[:, 0],
            a=action,
        )
        output["OBS0_ACT"] = cast_to_video_grid(self.world_model.decode_image(h_z))
        output["OBS0_ACT_REWARD"] = self.world_model.decode_reward(h_z).mode()

        # ---- First observation only conditioning ----

        h_z, _ = self.world_model.imagine_sequence(
            time_steps=action.shape[1],
            actor=self.actor,
            o=obs_feat[:, 0],
        )
        output["OBS0"] = cast_to_video_grid(self.world_model.decode_image(h_z))
        output["OBS0_REWARD"] = self.world_model.decode_reward(h_z).mode()

        # ---- No conditioning ----

        h_z, _ = self.world_model.imagine_sequence(
            time_steps=action.shape[1],
            batch_size=action.shape[0],
            actor=self.actor,
        )
        output["NONE"] = cast_to_video_grid(self.world_model.decode_image(h_z))
        output["NONE_REWARD"] = self.world_model.decode_reward(h_z).mode()

        return output


class WorldModel(nn.Module):
    """
    Recurrent State Space Model:

    o_t ----------------------------------------> [1, 2] -> linear -> z_t{posterior}
    [z_{t-1}, a_{t-1}] -> linear -> gru -> h_t ------/
                                        -> linear -> z_t{prior}
    """

    def __init__(self, action_dim=15, inner_dim=512):
        super().__init__()

        self.encoder = ImageEncoderResnet(in_channels=3, in_width=64)

        self.o_dim = self.encoder.out_features
        self.h_dim = 512
        self.z_num_bins = 32
        self.z_num_vars = 32
        self.z_dim = self.z_num_bins * self.z_num_vars
        self.h_z_dim = self.h_dim + self.z_dim

        self.h_0_hat = nn.Parameter(torch.zeros(self.h_dim))

        self.gru = nn.GRUCell(inner_dim, self.h_dim)
        self.proj_imagine_in = MLP(self.z_dim + action_dim, [inner_dim])
        self.proj_imagine_out = MLP(self.h_dim, [inner_dim], self.z_dim)
        self.proj_observe = MLP(self.h_dim + self.o_dim, [inner_dim], self.z_dim)
        self.reward_head = MLP(self.h_z_dim, [inner_dim] * 1, 255)
        self.cont_head = MLP(self.h_z_dim, [inner_dim] * 1, 1)

        self.image_head = ImageDecoderResnet(
            in_features=self.h_z_dim, out_width=64, out_channels=3
        )

        self.reward_head[-1].weight.data.zero_()  # type: ignore

    def make_latent_stochastic(self, z_logits: Tensor) -> distr.FlatMultivariateOneHot:
        z_logits = z_logits.unflatten(-1, (self.z_num_vars, self.z_num_bins))
        return make_unimix_one_hot(z_logits)

    def get_prior(self, h: Tensor) -> distr.FlatMultivariateOneHot:
        z_logits = self.proj_imagine_out(h)
        return self.make_latent_stochastic(z_logits)

    def get_posterior(self, h: Tensor, o: Tensor) -> distr.FlatMultivariateOneHot:
        h_o = torch.cat([h, o], dim=-1)
        z_logits = self.proj_observe(h_o)
        return self.make_latent_stochastic(z_logits)

    def forward(self, obs: Dict[str, Tensor]):
        # TODO: normalise image with +/- 0.5
        image = obs["rgb"].permute(0, 3, 1, 2).float() / 255.0
        o = self.encoder(image)
        if "in_state_h" in obs:
            h = obs["in_state_h"]
            h[obs["in_reset"]] = torch.tanh(self.h_0_hat).unsqueeze(0)
        else:
            h = torch.tanh(self.h_0_hat).expand(o.shape[0], -1)
        z_post = self.get_posterior(h=h, o=o)
        return h, z_post

    def dynamics_step(self, h: Tensor, z: Tensor, a: Tensor):
        x = self.proj_imagine_in(torch.cat([z, a], dim=-1))
        h = self.gru(x, h)
        return h

    def observe_sequence(
        self, o: Tensor, a: Tensor
    ) -> Tuple[Tensor, distr.FlatMultivariateOneHot]:
        assert (
            o.dim() == 3 and a.dim() == 3
        ), f"Expected 3D tensors, got {o.dim()} and {a.dim()}"
        assert (
            o.shape[0] == a.shape[0]
        ), f"Expected batch size to match, got {o.shape[0]} and {a.shape[0]}"
        assert (
            o.shape[1] == a.shape[1] + 1
        ), f"Expected {a.shape[1] + 1} observations, got {o.shape[1]}"

        h = torch.tanh(self.h_0_hat).expand(o.shape[0], -1)
        z_post = self.get_posterior(h, o[:, 0])
        z = z_post.mode()
        outputs: List[Tuple[Tensor, Tensor, Tensor]] = [(h, z, z_post.probs)]
        for t in range(o.shape[1] - 1):
            h = self.dynamics_step(h=h, z=z, a=a[:, t])
            z_post = self.get_posterior(h=h, o=o[:, t + 1])
            z = z_post.sample()
            outputs.append((h, z, z_post.probs))

        h = torch.stack([x[0] for x in outputs], dim=1)  # [B, T, h_dim]
        z = torch.stack([x[1] for x in outputs], dim=1)  # [B, T, z_dim]
        h_z = torch.cat((h, z), dim=-1)  # [B, T, h_dim + z_dim]
        z_post = distr.FlatMultivariateOneHot(
            torch.stack([x[2] for x in outputs], dim=1)
        )
        return h_z, z_post

    def imagine_sequence(
        self,
        time_steps: int,
        actor: "Actor",
        initial_h_z: Optional[Tensor] = None,
        o: Optional[Tensor] = None,
        a: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Imagine a sequence of observations and actions.
        """
        if initial_h_z is not None:
            # Initialize from h_z
            assert o is None
            h_z = initial_h_z
            h, z = initial_h_z.split([self.h_dim, self.z_dim], dim=-1)
            batch_size = h_z.shape[0]
        elif o is not None:
            # Initialize from observation
            h = torch.tanh(self.h_0_hat).expand(o.shape[0], -1)
            z = self.get_posterior(h, o).sample()
            h_z = torch.cat([h, z], dim=-1)
        else:
            assert batch_size is not None
            # Initialize randomly
            h = torch.tanh(self.h_0_hat).expand(batch_size, -1)
            z = self.get_prior(h).sample()
            h_z = torch.cat([h, z], dim=-1)

        h_z_list: List[Tensor] = [h_z]
        action_list: List[Tensor] = []
        for t in range(time_steps - 1):
            if a is not None:
                action_sample = a[:, t]
            else:
                action_distr = actor.forward(h_z)
                action_sample = action_distr.sample().detach()
            action_list.append(action_sample)

            h = self.dynamics_step(h=h, z=z, a=action_sample)
            z = self.get_prior(h).sample()
            h_z_list.append(h_z := torch.cat([h, z], dim=-1))

        action_sample = torch.stack(action_list, dim=1)
        h_z = torch.stack(h_z_list, dim=1)
        return h_z, action_sample

    def decode_reward(self, h_z: Tensor) -> distr.SymlogCategorical:
        # offset by 1 because reward_hat[:, 0] hasn't seen the first action yet
        logits = self.reward_head(h_z[:, 1:])
        return distr.SymlogCategorical(logits=logits)

    def decode_cont(self, h_z: Tensor) -> distr.Bernoulli:
        """
        Returns a logit that is negative if episode has terminated
        after the current time step. Note that prediction has T-1
        time steps.
        """
        return distr.Bernoulli(logit=self.cont_head(h_z[:, 1:]).squeeze(-1))

    def decode_image(self, h_z: Tensor) -> Tensor:
        """
        Returns an image reconstruction of the given
        states.
        """
        return self.image_head(h_z)

    @compile()
    def compute_loss(self, batch: Dict[str, Tensor]):
        """
        Computes world model losses.
        """
        image = batch["rgb"].permute(0, 1, 4, 2, 3).float() / 255.0
        action = F.one_hot(batch["action"], num_classes=15).float()
        weight = batch["valid"].float()

        obs_feat = self.encoder.forward(image)  # type: ignore
        h_z, z_post = self.observe_sequence(o=obs_feat, a=action[:, :-1])
        z_prior = self.get_prior(h_z[..., : self.h_dim])

        image_hat = self.decode_image(h_z)
        reward_hat = self.decode_reward(h_z)
        cont_hat = self.decode_cont(h_z)

        loss_image = F.mse_loss(image_hat, image, reduction="none").sum(
            dim=(-1, -2, -3)
        )
        loss_reward = -distr.log_prob(reward_hat, batch["reward"][:, :-1])
        loss_cont = -distr.log_prob(cont_hat, ~batch["is_done"][:, :-1])

        loss_image = torch.mean(loss_image * weight)
        loss_reward = torch.mean(loss_reward * weight[:, :-1])
        loss_cont = torch.mean(loss_cont * weight[:, :-1])

        loss_model = loss_image + loss_reward + loss_cont
        loss_dyn = z_post.detach().kl_div(z_prior).clamp(min=1.0).mean()
        loss_repr = z_post.kl_div(z_prior.detach()).clamp(min=1.0).mean()
        loss_wm = loss_model + 0.5 * loss_dyn + 0.1 * loss_repr

        losses = {
            "loss_cont": loss_cont,
            "loss_dyn": loss_dyn,
            "loss_image": loss_image,
            "loss_model": loss_model,
            "loss_repr": loss_repr,
            "loss_reward": loss_reward,
            "loss_wm": loss_wm,
        }

        return losses, h_z


class Actor(nn.Module):
    def __init__(self, in_features: int, inner_dim: int = 512, action_dim: int = 15):
        super().__init__()
        self.action_head = MLP(in_features, [inner_dim] * 2, action_dim)

    def forward(self, x: Tensor):
        logits = self.action_head(x)
        return make_unimix_one_hot(logits.unsqueeze(-2))


class Critic(nn.Module):
    def __init__(self, in_features: int, inner_dim: int = 512):
        super().__init__()
        self.horizon = 100
        self.gamma = 1.0 - 1.0 / self.horizon
        self.return_lambda = 0.95
        self.decay = 0.99
        self.slow_critic_decay = 0.98
        self.value_head = MLP(in_features, [inner_dim] * 2, 255)
        self.value_head[-1].weight.data.zero_()  # type: ignore

        self.slow_value_head = copy.deepcopy(self.value_head)
        self.slow_value_head.requires_grad_(False)

        self.register_buffer("scale_ema_low", torch.zeros(()))
        self.register_buffer("scale_ema_high", torch.zeros(()))
        self.register_buffer(
            "scale_quantiles", torch.tensor([0.05, 0.95]), persistent=False
        )

    def forward(self, x: Tensor):
        return distr.SymlogCategorical(logits=self.value_head(x))

    def slow_forward(self, x: Tensor):
        return distr.SymlogCategorical(logits=self.slow_value_head(x))

    def compute_returns(self, h_z: Tensor, world_model: WorldModel):
        rew = distr.mode(world_model.decode_reward(h_z))  # no reward for last frame
        cont = distr.mode(world_model.decode_cont(h_z))

        val = self.slow_forward(h_z).mean()
        discount = cont.float() * self.gamma
        lmbd = self.return_lambda

        # TODO: investigate difference to GAE
        ret_list = []
        next_val = val[:, -1]
        for t in reversed(range(val.shape[1] - 1)):
            target_val = (1.0 - lmbd) * val[:, t + 1] + lmbd * next_val
            ret_list.append(next_val := (rew[:, t] + discount[:, t] * target_val))

        ret = torch.stack(ret_list[::-1], dim=1)
        self.update_value_scale(ret)
        normed_adv = (ret - val[:, :-1]) / self.value_scale()
        return ret, normed_adv

    def update_value_scale(self, ret):
        low, high = torch.quantile(ret, q=self.scale_quantiles).detach().unbind(0)  # type: ignore
        self.scale_ema_low = self.decay * self.scale_ema_low + (1 - self.decay) * low
        self.scale_ema_high = self.decay * self.scale_ema_high + (1 - self.decay) * high

    def value_scale(self) -> Tensor:
        return torch.clamp(self.scale_ema_high - self.scale_ema_low, min=1.0)

    def update_slow_critic(self):
        for param, slow_param in zip(
            self.value_head.parameters(), self.slow_value_head.parameters()
        ):
            slow_param.data.lerp_(param.data, 1.0 - self.slow_critic_decay)


def make_unimix_one_hot(logits: Tensor, alpha: float = 0.01):
    """
    Returns a one hot distribution that is an interpolation
    between a uniform distribution and a categorical distribution
    with logits `logits`, proposed in the DreamerV3 paper.
    """
    probs = torch.softmax(logits, dim=-1)
    if alpha > 0.0:
        probs = probs * (1 - alpha) + alpha / probs.shape[-1]
    return distr.FlatMultivariateOneHot(probs=probs)


def cast_to_video_grid(x: Tensor) -> Tensor:
    """
    Returns video in the shape [T, H, B * W, C]
    for visualisation purposes.
    """
    # [B, T, C, H, W] -> [T, H, B * W, C]
    if torch.is_floating_point(x):
        x = (x * 255.0).clamp(min=0.0, max=255.0)
        x = x.byte()
    return x.permute(1, 3, 0, 4, 2).flatten(-3, -2)


"""

---- Todo: ----


---- Notes: ----

Reward and value are encoded through 255 bins:

y = symexp( sum_i[ p[i] * (i / (n-1) * (max - min) + min) ] )

where i is the bin index and p[i] is the probability of that bin
as predicted by the model. p[i] is supervised using two-hot encoded
(symlog(r) - min) / (max - min) * n.

Episode ends probs are represented as sigmoids. The loss is
`softplus(torch.where(ground_truth, -logits, logits))`, i.e.
-log(p) if ground_truth is true and -log(1-p) otherwise.

Critic and actor losses are future discounted.

w_0 = 1.0
w_t = w_{t-1} * discount * (1-p(continue))


!! Imagination starts from every possible state in the sequence,
not just the last frame in the sequence. !!

!! Critic and reward get initialised to zero weights for last layer !!
=> Mean will be zero, hence no overestimation / underestimation.


DreamerV3 uses post-norm MLPs, i.e. matmul -> norm -> activation, instead of the
usual norm -> matmul -> activation.


---- Rewards ----

PPO:
A_t := r_t + γ * (v_{t+1} - v_t + λ * A_{t+1})

->  A'_t := r_t + γ * (v_{t+1} - v_t + λ * A'_{t+1}) + v_t
->  A'_t := r_t + γ * (v_{t+1} + λ * A_{t+1}) + v_t


DreamerV3:
R_t := r_t + γ * ((1.0 - λ) * v_{t+1} + λ * R_{t+1})


"""
