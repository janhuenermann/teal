from typing import Dict, NamedTuple, Protocol
import numpy as np

from torch import Tensor, nn


class AgentToTrainerParams(NamedTuple):
    inference_steps: int
    learn_steps: int


class Agent(Protocol):
    trainer_params: AgentToTrainerParams
    policy: nn.Module

    def inference(self, observation: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Returns the action to take in the environment and any additional
        information that should be stored in the buffer.
        """
        ...

    def add_transition(self, transition: Dict[str, np.ndarray]): ...

    def should_learn(self) -> bool:
        return True

    def sample_batch(self) -> Dict[str, np.ndarray]: ...

    def learn_on_batch(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]: ...
