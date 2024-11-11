"""IQL-SAC algorithm implementation."""

import torch
import torch.optim as optim

from robin.rl.algorithms.actors import ActorSAC
from robin.rl.algorithms.critics import Critic
from robin.rl.entities import StatsSubprocVectorEnv


class IQLSAC:
    """
    Independent Q-Learning Soft Actor-Critic.
    
    Attributes:
        num_agents (int): Number of agents in the environment.
        actor (list[Actor]): List of actor networks for each agent.
        qf1 (list[Critic]): List of the first Q networks for each agent.
        qf2 (list[Critic]): List of the second Q networks for each agent.
        qf1_target (list[Critic]): List of target Q networks for each agent.
        qf2_target (list[Critic]): List of target Q networks for each agent.
        q_optimizer (list[torch.optim]): List of Q network optimizers for each agent.
        actor_optimizer (list[torch.optim]): List of actor network optimizers for each agent.
    """

    def __init__(self, env: StatsSubprocVectorEnv, device: torch.device, policy_lr: float = 3e-4, q_lr: float = 1e-3) -> None:
        """
        Initialize the Independent Q-Learning Soft Actor-Critic algorithm.
        
        Each agent has its own actor, Q networks, and target Q networks.
        
        Args:
            env (StatsSubprocVectorEnv): The environment to train on.
            device (torch.device): The device to run the algorithm on.
            policy_lr (float): The learning rate of the policy network optimizer.
            q_lr (float): The learning rate of the Q network optimizer.
        """
        self.num_agents = env.get_env_attr('num_agents')[0]
        self.actor = [ActorSAC(env, agent_idx).to(device) for agent_idx in range(self.num_agents)]
        self.qf1 = [Critic(env, agent_idx).to(device) for agent_idx in range(self.num_agents)]
        self.qf2 = [Critic(env, agent_idx).to(device) for agent_idx in range(self.num_agents)]
        self.qf1_target = [Critic(env, agent_idx).to(device) for agent_idx in range(self.num_agents)]
        self.qf2_target = [Critic(env, agent_idx).to(device) for agent_idx in range(self.num_agents)]
        for agent_idx in range(self.num_agents):
            self.qf1_target[agent_idx].load_state_dict(self.qf1[agent_idx].state_dict())
            self.qf2_target[agent_idx].load_state_dict(self.qf2[agent_idx].state_dict())
        self.q_optimizer = [optim.Adam(list(self.qf1[i].parameters()) + list(self.qf2[i].parameters()), lr=q_lr) for i in range(self.num_agents)]
        self.actor_optimizer = [optim.Adam(list(self.actor[i].parameters()), lr=policy_lr) for i in range(self.num_agents)]

    def get_action(self, obs: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
        """
        Get the actions for each agent in the environment.
        
        Args:
            obs (torch.Tensor): The observations for each agent.
            
        Returns:
            actions (list[torch.Tensor]): The actions for each agent.
            log_prob (torch.Tensor): The log probabilities of the actions.
            mean (list[torch.Tensor]): The means of the actions.
        """
        actions, log_prob, mean = zip(*[actor.get_action(obs[agent_idx]) for agent_idx, actor in enumerate(self.actor)])
        return list(actions), torch.stack(log_prob), list(mean)
