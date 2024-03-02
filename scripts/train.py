import hydra
from omegaconf import DictConfig, OmegaConf

from teal.callbacks.checkpointing import ExportCallback
from teal.callbacks.evaluation import (
    EvaluationCallback,
)
from teal.trainer import Trainer


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    print("")
    print("========== ==========")
    print("")
    print(OmegaConf.to_yaml(cfg))

    trainer: Trainer = hydra.utils.instantiate(cfg)
    trainer.register_callbacks(
        [ExportCallback(every_nth=20), EvaluationCallback(every_nth=20)]
    )
    trainer.run(cfg)


if __name__ == "__main__":
    main()
