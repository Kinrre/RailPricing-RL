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

    def eval(self) -> None:
        """
        Set the model to evaluation mode.
        """
        for agent in self.actor:
            agent.eval()

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

    @classmethod
    def load_model(cls, path: str, env: StatsSubprocVectorEnv, device: torch.device) -> 'IQLSAC':
        """
        Load the model parameters from a file.

        Args:
            path (str): The path to load the model parameters from.
            env (StatsSubprocVectorEnv): The environment.
            device (torch.device): The device to run the algorithm on.
        """
        save_dict = torch.load(path)
        model = cls(env, device)
        model.num_agents = save_dict['num_agents']
        for agent_idx in range(model.num_agents):
            model.actor[agent_idx].load_state_dict(save_dict['actor'][agent_idx])
            model.qf1[agent_idx].load_state_dict(save_dict['qf1'][agent_idx])
            model.qf2[agent_idx].load_state_dict(save_dict['qf2'][agent_idx])
            model.qf1_target[agent_idx].load_state_dict(save_dict['qf1_target'][agent_idx])
            model.qf2_target[agent_idx].load_state_dict(save_dict['qf2_target'][agent_idx])
            model.q_optimizer[agent_idx].load_state_dict(save_dict['q_optimizer'][agent_idx])
            model.actor_optimizer[agent_idx].load_state_dict(save_dict['actor_optimizer'][agent_idx])
        return model

    def save_model(self, path: str) -> None:
        """
        Save the model parameters to a file.
        
        Args:
            path (str): The path to save the model parameters to.
        """
        save_dict = {
            'num_agents': self.num_agents,
            'actor': [actor.state_dict() for actor in self.actor],
            'qf1': [qf1.state_dict() for qf1 in self.qf1],
            'qf2': [qf2.state_dict() for qf2 in self.qf2],
            'qf1_target': [qf1_target.state_dict() for qf1_target in self.qf1_target],
            'qf2_target': [qf2_target.state_dict() for qf2_target in self.qf2_target],
            'q_optimizer': [q_opt.state_dict() for q_opt in self.q_optimizer],
            'actor_optimizer': [actor_opt.state_dict() for actor_opt in self.actor_optimizer]
        }
        torch.save(save_dict, path)
