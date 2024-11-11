"""Critics for the RL algorithms."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from robin.rl.entities import StatsSubprocVectorEnv
from robin.rl.algorithms.constants import HIDDEN_SIZE


class Critic(nn.Module):
    """
    Critic network.
    
    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer.
    """

    def __init__(self, env: StatsSubprocVectorEnv, agent_idx: int) -> None:
        """
        Initialize the Critic network.
        
        Args:
            env (StatsSubprocVectorEnv): The environment to train on.
            agent_idx (int): The index of the agent in the environment.
        """
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.observation_space[0][agent_idx].shape).prod() + np.prod(env.action_space[0][agent_idx].shape),
            HIDDEN_SIZE
        )
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass of the Critic network.
        
        Args:
            x (torch.Tensor): The input tensor.
            action (torch.Tensor): The action tensor.
        """
        x = torch.cat([x, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
