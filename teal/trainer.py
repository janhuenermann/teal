import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence
import numpy as np
from omegaconf import DictConfig
import torch
from tqdm import trange

from teal.env import Env
from teal.agent import Agent
from teal.logging import Logger, MetricsBucket
from teal.utils.timings import print_timings_summary, timed_scope

IS_DEBUG = os.getenv("DEBUG", "0") == "1"
SHOULD_COMPILE = os.getenv("COMPILE", "1") == "1" and sys.version_info < (3, 11)


class Trainer:
    def __init__(
        self,
        env: Env,
        agent: Agent,
        logger: Logger,
        log_dir: str,
        log_every_nth: int = 1,
        print_timings_every_nth: int = 50,
        max_steps: int = 20_000,
    ):
        self.env = env
        self.agent = agent
        self.logger = logger
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.inference_steps = 0
        self.learn_steps = 0
        self.max_steps = max_steps
        self.metrics = MetricsBucket()

        self.callbacks: List[TrainerCallback] = [  # type: ignore
            self.agent,  # type: ignore
            LogCommitCallback(every_nth=log_every_nth),
            PrintTimingsCallback(every_nth=print_timings_every_nth),
        ]

        trainer_params = self.agent.trainer_params
        self.inference_frequency = trainer_params.inference_steps
        self.learn_frequency = trainer_params.learn_steps

        self.device = find_system_device()
        self.agent.to(self.device)  # type: ignore

        self.policy_script = torch.jit.script(self.agent.policy)  # type: ignore
        self.policy_state = {}

    def run(self, cfg: DictConfig):
        print("========== ==========")
        print("Starting new training run 🏃....")

        self.config = cfg
        self.logger.store_config(cfg)

        torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore
        torch.backends.cudnn.benchmark = True  # type: ignore
        torch.set_float32_matmul_precision("high")

        while self.global_step < self.max_steps:
            self.invoke("on_before_global_step")

            with timed_scope("global_step"):
                self.inference_loop()
                self.learn_loop()

            self.global_step += 1
            self.invoke("on_after_global_step")

        self.invoke("on_training_end")
        print("Bye 👋")

    @torch.no_grad()
    def inference_loop(self):
        self.invoke("on_before_inference_phase")
        for _ in trange(
            self.inference_frequency, ncols=90, desc="Inference", disable=not IS_DEBUG
        ):
            obs = self.env.observe()
            th_obs = _cast_dict_to_tensors(obs, self.device)
            th_obs.update({f"in_{k}": v for k, v in self.policy_state.items()})
            with timed_scope("inference"):
                th_output = self.agent.inference(th_obs)

            outputs = _cast_tensors_to_dict(th_output)
            with timed_scope("env_step"):
                reward, is_done = self.env.act(outputs["action"])

            transition = dict(**obs, **outputs, reward=reward, is_done=is_done)
            self.agent.add_transition(transition)
            self.policy_state["reset"] = _cast_array_to_tensor(is_done, self.device)
            self.policy_state.update(
                {k: v for k, v in th_output.items() if k.startswith("state")}
            )
            self.inference_steps += 1

    def learn_loop(self):
        if not self.agent.should_learn():
            return
        self.invoke("on_before_learn_phase")
        for _ in trange(
            self.learn_frequency, ncols=90, desc="Learning", disable=not IS_DEBUG
        ):
            self.invoke("on_before_learn_step")

            with timed_scope("fetch_batch"):
                batch = self.agent.sample_batch()

            with timed_scope("learn_on_batch", on_gpu=True):
                th_batch = _cast_dict_to_tensors(batch, self.device)
                th_outputs = self.agent.learn_on_batch(th_batch)

            outputs = _cast_tensors_to_dict(th_outputs)
            self.metrics.log(**outputs)

            self.learn_steps += 1
            self.invoke("on_after_learn_step")
        self.invoke("on_after_learn_phase")

    def register_callback(self, callback: "TrainerCallback"):
        self.callbacks.append(callback)

    def register_callbacks(self, callbacks: Sequence["TrainerCallback"]):
        self.callbacks.extend(callbacks)

    def invoke(self, name: str, **kwargs):
        for callback in self.callbacks:
            if not hasattr(callback, name):
                continue
            fn = getattr(callback, name)
            fn(trainer=self, **kwargs)


class TrainerCallback:
    def on_before_global_step(self, trainer: Trainer): ...

    def on_before_inference_phase(self, trainer: Trainer): ...

    def on_before_learn_phase(self, trainer: Trainer): ...

    def on_before_learn_step(self, trainer: Trainer): ...

    def on_after_learn_step(self, trainer: Trainer): ...

    def on_after_learn_phase(self, trainer: Trainer): ...

    def on_after_global_step(self, trainer: Trainer): ...

    def on_training_end(self, trainer: Trainer): ...


class EveryNthCallback(TrainerCallback):
    def __init__(self, every_nth: int, run_on_first_step: bool = True):
        self.every_nth = every_nth
        self.run_on_first_step = run_on_first_step

    def run(self, trainer: Trainer):
        raise NotImplementedError

    def should_trigger(self, trainer: Trainer):
        return trainer.global_step % self.every_nth == 0 or (
            self.run_on_first_step and trainer.global_step == 1
        )

    def on_after_global_step(self, trainer: Trainer):
        if self.should_trigger(trainer):
            self.run(trainer)

    def on_training_end(self, trainer: Trainer):
        self.run(trainer)


class LogCommitCallback(EveryNthCallback):
    def __init__(self, every_nth: int):
        super().__init__(every_nth, run_on_first_step=False)

    def run(self, trainer: Trainer):
        trainer.logger.commit(trainer.metrics, global_step=trainer.global_step)


class PrintTimingsCallback(EveryNthCallback):
    def __init__(self, every_nth: int):
        super().__init__(every_nth, run_on_first_step=True)

    def should_trigger(self, trainer: Trainer):
        # Also print timings on the second step, because first step is usually warmup
        return super().should_trigger(trainer) or trainer.global_step == 2

    def run(self, trainer: Trainer):
        print_timings_summary(step=trainer.global_step)


def _cast_tensor_to_array(data: torch.Tensor):
    return data.detach().cpu().numpy()


def _cast_tensors_to_dict(data: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    return {k: _cast_tensor_to_array(v) for k, v in data.items()}


def _cast_array_to_tensor(data: np.ndarray, device: torch.device):
    return torch.from_numpy(data).to(device)


def _cast_dict_to_tensors(
    data: Dict[str, np.ndarray], device: torch.device
) -> Dict[str, torch.Tensor]:
    return {k: _cast_array_to_tensor(v, device) for k, v in data.items()}


def cast_dict_to_system_device(data: Dict[str, np.ndarray]):
    return _cast_dict_to_tensors(data, find_system_device())


def cast_dict_to_numpy(data: Dict[str, torch.Tensor]):
    return _cast_tensors_to_dict(data)


def find_system_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():  # type: ignore
        return torch.device("mps:0")
    else:
        return torch.device("cpu")


def compile(fullgraph=True, **kwargs):
    """
    Compiles the pytorch function with `torch.compile`
    unless the environment variable COMPILE is set to 0.
    """

    def decorator(fn):
        if not SHOULD_COMPILE:
            return fn

        return torch.compile(fn, fullgraph=fullgraph, **kwargs)

    return decorator
