import logging
from teal.trainer import EveryNthCallback, Trainer


class ExportCallback(EveryNthCallback):
    def __init__(self, every_nth: int):
        super().__init__(every_nth, run_on_first_step=True)

    def run(self, trainer: Trainer):
        logging.info("Exporting policy...")
        policy_path = trainer.log_dir / "policy.torchscript"
        trainer.policy_script.save(policy_path)  # type: ignore
        logging.info(f"Saved policy to {policy_path}")
