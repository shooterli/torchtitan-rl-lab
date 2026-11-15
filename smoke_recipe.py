"""Qwen3-0.6B rollout scheduling smoke test for TorchTitan v0.4.0."""

from torchtitan.rl.controller.controller import Controller
from torchtitan.rl.generator.vllm.stat_logger import VllmOtelStatLogger
from torchtitan_recipes.rl.alphabet_sort import rl_grpo_qwen3_0_6b_varlen


def _smoke(target_offpolicy_steps: int) -> Controller.Config:
    config = rl_grpo_qwen3_0_6b_varlen()
    config.async_loop.target_offpolicy_steps = target_offpolicy_steps
    config.async_loop.validation.steps = 0
    config.rollouter.training_dataloader.dataset.max_turns = 1
    config.rollouter.validation_dataset.max_turns = 1
    config.generator.engine.stat_logger = VllmOtelStatLogger.Config()
    config.metrics.enable_wandb = False
    config.metrics.console_log_keys_train = [
        "timing/step/total",
        "perf/trainer/tokens_per_second_full_step",
        "train_batch/policy_age",
        "rollout_reward/_mean",
    ]
    return config


def sync_smoke() -> Controller.Config:
    return _smoke(target_offpolicy_steps=0)


def async_smoke() -> Controller.Config:
    return _smoke(target_offpolicy_steps=1)


def sync_fixed_cap() -> Controller.Config:
    return _fixed_cap(target_offpolicy_steps=0)


def async_fixed_cap() -> Controller.Config:
    return _fixed_cap(target_offpolicy_steps=1)


def _fixed_cap(target_offpolicy_steps: int) -> Controller.Config:
    config = _smoke(target_offpolicy_steps)
    config.generator_max_num_seqs = 128
    return config
