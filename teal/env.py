from typing import Dict, Protocol, Tuple

from procgen import ProcgenGym3Env
import numpy as np


class Env(Protocol):
    def observe(self) -> Dict[str, np.ndarray]:
        """
        Returns current observation.
        """
        ...

    def act(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (reward, is_done)
        """
        ...


class ProcgenEnv:
    def __init__(self, env_name: str, batch_size: int = 32):
        self._env = ProcgenGym3Env(
            num=batch_size,
            env_name=env_name,
            distribution_mode="hard",
            use_backgrounds=False,
            restrict_themes=True,
        )
        self._next_obs = None

    def observe(self):
        if self._next_obs is None:
            self._next_obs = self._env.observe()[1]
        return self._next_obs

    def act(self, action: np.ndarray):
        self._env.act(action)  # type: ignore
        reward, self._next_obs, next_is_first = self._env.observe()
        is_last = next_is_first  # current is last if next is first
        return reward, is_last
