"""VDN algorithm implementation."""

import torch
import torch.nn as nn

from robin.rl.entities import StatsSubprocVectorEnv
from robin.rl.algorithms.iql_sac import IQLSAC


class VDNMixer(nn.Module):
    """
    Value Decomposition Network (VDN) mixer.
    """
    
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, q_values: torch.Tensor) -> torch.Tensor:
        """
        Sum the Q values for each agent.
        
        Args:
            q_values (torch.Tensor): The Q values for each agent.
            
        Returns:
            torch.Tensor: The summed Q values.
        """
        return torch.sum(q_values, dim=0)

class VDN(IQLSAC):
    """
    Value Decomposition Network (VDN) algorithm.
    
    Attributes:
        vdn_mixer (VDNMixer): The VDN mixer network.
    """

    def __init__(self, env: StatsSubprocVectorEnv, device: torch.device, policy_lr: float = 3e-4, q_lr: float = 1e-3) -> None:
        super().__init__(env, device, policy_lr, q_lr)
        self.vdn_mixer = VDNMixer().to(device)
