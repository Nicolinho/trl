import os
from dataclasses import dataclass

from ..trainer.utils import OnPolicyConfig


@dataclass
class RLOOConfig(OnPolicyConfig):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    reward_model_path: str = "EleutherAI/pythia-160m"
    """the path to the reward model"""

    # ppo config
    num_ppo_epochs: int = 4
    """the number of epochs to train"""
    whiten_rewards: bool = False
    """whether to whiten the rewards"""
    kl_coef: float = 0.05
    """the KL coefficient"""
    entropy_coef: float = 0.
    entropy_loss: int = 0   # whether to use the entropy loss or not ( set 1 or 0)
    """the KL coefficient"""
    cliprange: float = 0.2
    """the clip range"""

    risk_aware_coeff_start: float = 5.
    risk_aware_coeff_end: float = 5.
    risk_aware_loss: int = 0 # whether to use the risk aware loss or not ( set 1 or 0)

    reward_scale: float = 1.
    reward_bias: float = 0.
    """scaling and bias of the reward from the reward model"""

    # rloo config
    rloo_k: int = 2
    """REINFORCE Leave-One-Out (RLOO) number of online samples per prompt"""