import logging
from hydra.utils import instantiate
import numpy as np

from teal.trainer import EveryNthCallback, Trainer
from teal.utils.evaluation import evaluate_policy


class EvaluationCallback(EveryNthCallback):
    def __init__(self, every_nth: int):
        super().__init__(every_nth, run_on_first_step=True)

    def run(self, trainer: Trainer):
        logging.info("Evaluating policy...")
        env = instantiate(trainer.config.env)
        episode_returns = evaluate_policy(trainer.policy_script, env=env)  # type: ignore
        median_reward = np.median(episode_returns)
        avg_reward = episode_returns.mean()
        std_reward = episode_returns.std()
        eval_metrics = {
            "eval/median_reward": median_reward,
            "eval/avg_reward": avg_reward,
            "eval/std_reward": std_reward,
        }
        trainer.metrics.log_dict(eval_metrics)
        logging.info(f"Eval: At step {trainer.global_step}")
        logging.info(
            f"Eval: Average reward: {avg_reward:.4f} +- {std_reward:.4f} over {len(episode_returns)} episodes"
        )
