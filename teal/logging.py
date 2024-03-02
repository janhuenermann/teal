from collections import defaultdict
import os
from pathlib import Path
import time
from typing import Any, Dict, Protocol, Union
import numpy as np
from omegaconf import OmegaConf, DictConfig
import torch

import wandb

from teal.utils.video import VideoWriter


class MetricsBucket:
    def __init__(self):
        self.data = defaultdict(list)

    def log(self, **kwargs):
        self.log_dict(kwargs)

    def log_dict(self, data: Dict[str, Any]):
        """
        Adds the given values to the bucket.
        """
        for key, value in data.items():
            if torch.is_tensor(value):
                assert (
                    value.ndim == 0
                ), f"Expected scalar, got tensor with shape {value.shape}"
                value = value.detach().cpu().item()
            elif isinstance(value, (np.generic, np.ndarray)):
                assert (
                    value.ndim == 0
                ), f"Expected scalar, got array with shape {value.shape}"
                value = value.item()
            elif isinstance(value, (float, int)):
                pass
            else:
                raise ValueError(f"Expected scalar, got {type(value)}")
            self.data[key].append(value)

    def reduce(self) -> Dict[str, Union[float, int]]:
        """
        Returns the average value for each key in the bucket.
        """
        return {key: np.mean(values) for key, values in self.data.items()}  # type: ignore

    def squash(self):
        """
        Reduces the data in the bucket to a single value per key and
        clears the bucket.
        """
        result = self.reduce()
        self.clear()
        return result

    def clear(self):
        self.data.clear()


class Logger(Protocol):
    def store_config(self, cfg: DictConfig):
        """
        Saves the configuration to the logger.
        """

    def log_video(self, name: str, video: np.ndarray, fps: int):
        """
        Logs a video to the logger. Video needs to be THWC.
        """

    def commit(self, metrics: MetricsBucket, global_step: int):
        """
        Sends the metrics to the logger and clear the bucket.
        """


class NoopLogger(Logger):
    def __init__(self, project_name: str = ""):
        del project_name

    def commit(self, metrics: MetricsBucket, global_step: int):
        metrics.clear()

    def store_config(self, cfg: DictConfig):
        pass


class WandbLogger(Logger):
    def __init__(
        self,
        project_name: str,
        log_dir: str,
    ):
        os.makedirs(log_dir, exist_ok=True)
        self.videos: Dict[str, wandb.Video] = {}
        wandb.init(
            project=project_name,
            dir=log_dir,
        )

    def store_config(self, cfg: DictConfig):
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

    def log_video(self, key: str, video: np.ndarray, fps=5):
        assert wandb.run is not None
        video_name = time.strftime("%Y-%m-%d_%H-%M-%S") + "_" + key + ".mp4"
        video_path = Path(wandb.run.dir) / video_name
        with VideoWriter(video_path, fps=5) as writer:
            for frame in video:
                writer.write_frame(frame)
        self.videos[key] = wandb.Video(str(video_path), fps=fps, format="mp4")

    def commit(self, metrics: MetricsBucket, global_step: int):
        data: Dict[str, Any] = metrics.squash()
        data.update(global_step=global_step)
        data.update(self.videos)
        wandb.log(data)
        self.videos.clear()
