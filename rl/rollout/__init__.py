"""Self-hosted Rollout and Prime-RL Orchestrator Package."""

from rl.rollout.app import app
from rl.rollout.config import (
    BASE_MODEL,
    DEFAULT_INFERENCE_MODEL,
    PRIME_RL_REV,
    SFT_ADAPTER_PATH,
    VERIFIERS_REV,
)
from rl.rollout.evaluators import run_harbor_e2b_eval, run_verifiers_evaluation
from rl.rollout.inference import get_local_e2b_key, launch_prime_inference_server, wait_for_inference_ready
from rl.rollout.trainer import resume_prime_rl_training, run_prime_rl_training
