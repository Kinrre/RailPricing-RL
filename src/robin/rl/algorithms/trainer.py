"""Trainer module for experimenting with different RL algorithms."""

import datetime
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import time
import tyro
import yaml

from dataclasses import dataclass
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from typing import Literal, Union

from robin.rl.entities import RobinEnvFactory, StatsSubprocVectorEnv
from robin.rl.algorithms.buffers import ReplayBuffer
from robin.rl.algorithms.constants import IS_COOPERATIVE
from robin.rl.algorithms.iql_sac import IQLSAC
from robin.rl.algorithms.vdn import VDN


@dataclass
class TrainerArgs:
    algorithm: Literal['iql_sac', 'vdn'] = 'iql_sac'
    """the algorithm to use"""
    output_dir: str = 'models'
    """the output directory to store the logs"""
    exp_name: str = 'default'
    """the name of this experiment"""
    supply_config: str = 'configs/rl/supply_data_connecting.yml'
    """path to the supply data configuration file"""
    demand_config: str = 'configs/rl/demand_data_connecting.yml'
    """path to the demand data configuration file"""
    seed: int = 0
    """seed of the experiment"""
    cuda: bool = True
    """whether to use cuda"""
    n_workers: int = 16
    """number of workers for training (it should be the number of cores)"""
    total_timesteps: int = 1_000_000
    """total timesteps to train the agent"""
    buffer_size: int = 1_000_000
    """size of the replay buffer"""
    learning_starts: int = 50_000
    """number of timesteps to start learning"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    gamma: float = 0.99
    """discount factor"""
    tau: float = 0.005
    """soft update factor"""
    policy_lr: float = 3e-4
    """learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """learning rate of the q network optimizer"""
    alpha: float = 0.2
    """entropy regularization coefficient"""


class Trainer:
    """
    Trainer class for training the agent.
    
    Attributes:
        args (TrainerArgs): Arguments for the trainer.
        run_name (str): Name of the experiment.
        writer (SummaryWriter): Tensorboard writer.
        device (torch.device): Device to run the experiment on.
        env (StatsSubprocVectorEnv): ROBIN environment.
        agent (Union[IQLSAC | VDN]): The agent to train.
        replay_buffer (ReplayBuffer): Replay buffer for storing the experiences.
    """ 
    
    def __init__(self) -> None:
        """
        Initialize the trainer.
        
        It uses the arguments from the command line to initialize the trainer,
        please refer to the TrainerArgs class.
        """
        self.args = tyro.cli(TrainerArgs)
        now = datetime.datetime.now().strftime('%d%m%y-%H%M%S')
        self.run_name = f'{self.args.output_dir}/{self.args.exp_name}/{self.args.algorithm}/{self.args.seed}/{now}'
        self.writer = SummaryWriter(self.run_name)
        self.log_args_and_git_commit()
        self.set_seed(self.args.seed)
        self.device = torch.device('cuda' if self.args.cuda and torch.cuda.is_available() else 'cpu')
        self.env = self.create_env()
        self.agent: Union[IQLSAC | VDN] = self._get_agent(self.args.algorithm)(
            env=self.env,
            device=self.device,
            policy_lr=self.args.policy_lr,
            q_lr=self.args.q_lr
        )
        self.replay_buffer = ReplayBuffer(
            max_steps=self.args.buffer_size,
            num_agents=self.agent.num_agents,
            obs_dims=[obsp.shape[0] for obsp in self.env.observation_space[0]],
            ac_dims=[acsp.shape[0] for acsp in self.env.action_space[0]]
        )
    
    def _get_agent(self, algorithm: str) -> Union[IQLSAC | VDN]:
        """
        Get the agent based on the algorithm name.
        
        Args:
            algorithm (str): The algorithm to use.
            
        Returns:
            Union[IQLSAC | VDN]: The agent based on the algorithm name.
        """
        if algorithm == 'iql_sac':
            return IQLSAC
        elif algorithm == 'vdn':
            return VDN
        else:
            raise Exception(f'Algorithm {algorithm} not supported.')
    
    def _log_yaml(self, filename: str, data: dict, use_vars: bool = False) -> None:
        """
        Log the data to a yaml file.
        
        Args:
            filename (str): The filename to log the data.
            data (dict): The data to log.
            use_vars (bool): Whether to use the vars function to log the data.
        """
        with open(os.path.join(self.run_name, filename), 'w') as f:
            if use_vars:
                yaml.dump(vars(data), f, sort_keys=False)
            else:
                yaml.dump(data, f, sort_keys=False)

    def create_env(self) -> StatsSubprocVectorEnv:
        """
        Create the ROBIN environment.
            
        Returns:
            StatsSubprocVectorEnv: The ROBIN environment.
        """
        env_fns = [
            lambda: RobinEnvFactory.create(
                path_config_supply=self.args.supply_config,
                path_config_demand=self.args.demand_config,
                multi_agent=True,
                cooperative=IS_COOPERATIVE[self.args.algorithm],
                discrete_action_space=False,
                seed=self.args.seed + i * 1000
            ) for i in range(self.args.n_workers)
        ]
        env = StatsSubprocVectorEnv(env_fns=env_fns, log_dir=self.run_name)
        env.seed([self.args.seed + i * 1000 for i in range(self.args.n_workers)])
        return env

    def log_stats(
            self,
            global_step: int,
            start_time: float,
            i: int,
            qf1_a_values: torch.Tensor,
            qf2_a_values: torch.Tensor,
            qf1_loss: torch.Tensor,
            qf2_loss: torch.Tensor,
            qf_loss: torch.Tensor,
            actor_loss: torch.Tensor
    ) -> None:
        """
        Log the statistics of the training.
        
        Args:
            global_step (int): Current global step.
            start_time (float): Start time of the experiment.
            i (int): Index of the agent.
            qf1_a_values (torch.Tensor): First Q function values.
            qf2_a_values (torch.Tensor): Second Q function values.
        """
        if global_step % 100 == 0:
            self.writer.add_scalar(f'losses_{i}/qf1_values', qf1_a_values.mean().item(), global_step)
            self.writer.add_scalar(f'losses_{i}/qf2_values', qf2_a_values.mean().item(), global_step)
            self.writer.add_scalar(f'losses_{i}/qf1_loss', qf1_loss.item(), global_step)
            self.writer.add_scalar(f'losses_{i}/qf2_loss', qf2_loss.item(), global_step)
            self.writer.add_scalar(f'losses_{i}/qf_loss', qf_loss.item() / 2.0, global_step)
            self.writer.add_scalar(f'losses_{i}/actor_loss', actor_loss.item(), global_step)
            self.writer.add_scalar('losses/alpha', self.args.alpha, global_step)
            logger.info(f'SPS: {int(global_step / (time.time() - start_time))}')
            self.writer.add_scalar('charts/SPS', int(global_step / (time.time() - start_time)), global_step)

    def log_args_and_git_commit(self) -> None:
        """
        Log arguments and git commit to the log directory.
        """
        self._log_yaml('args.yml', self.args, use_vars=True)
        self._log_yaml('supply_config.yml', yaml.safe_load(open(self.args.supply_config)), use_vars=False)
        self._log_yaml('demand_config.yml', yaml.safe_load(open(self.args.demand_config)), use_vars=False)
        os.system(f'git log -1 > {self.run_name}/commit.txt')
        os.system(f'git diff > {self.run_name}/diff.patch')

    def train(self) -> None:
        """
        Train the agent.
        """
        start_time = time.time()
        obs, _ = self.env.reset()

        for global_step in range(0, self.args.total_timesteps, self.env.n_envs):
            # Sample actions from the agent and step the environment
            agent_actions = self.sample_actions(obs, global_step)
            next_obs, terminations = self.step_and_push(obs, agent_actions)
            
            # Update observations
            obs = next_obs
            if terminations.all():
                obs, _ = self.env.reset()

            # Critic training
            if global_step > self.args.learning_starts:
                self.train_critic(global_step, start_time)

        self.agent.save_model(f'{self.run_name}/model.pt')
        self.env.close()
        self.writer.close()
    
    def train_critic(self, global_step: int, start_time: float) -> None:
        """
        Train the critic networks.
        
        Args:
            global_step (int): Current global step.
            start_time (float): Start time of the experiment.
        """
        if self.args.algorithm == 'iql_sac':
            self.train_iql_sac(global_step, start_time)
        elif self.args.algorithm == 'vdn':
            self.train_vdn(global_step, start_time)
    
    def train_iql_sac(self, global_step: int, start_time: float) -> None:
        """
        Train the IQL-SAC agent.
        
        Args:
            global_step (int): Current global step.
            start_time (float): Start time of the experiment.
        """
        for agent_idx in range(self.agent.num_agents):
            # Sampling different batches for each agent to avoid overfitting
            observations, actions, rewards, next_observations, dones = self.replay_buffer.sample(self.args.batch_size, to_gpu=self.args.cuda)
            
            # Calculate the target Q values
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = self.agent.actor[agent_idx].get_action(next_observations[agent_idx])
                qf1_next_target = self.agent.qf1_target[agent_idx](next_observations[agent_idx], next_state_actions)
                qf2_next_target = self.agent.qf2_target[agent_idx](next_observations[agent_idx], next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.args.alpha * next_state_log_pi
                next_q_value = rewards[agent_idx] + (1 - dones[agent_idx]) * self.args.gamma * (min_qf_next_target).squeeze()
        
            qf1_a_values = self.agent.qf1[agent_idx](observations[agent_idx], actions[agent_idx]).squeeze()
            qf2_a_values = self.agent.qf2[agent_idx](observations[agent_idx], actions[agent_idx]).squeeze()
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # Optimize critic
            self.agent.q_optimizer[agent_idx].zero_grad()
            qf_loss.backward()
            self.agent.q_optimizer[agent_idx].step()
            
            # Calculate actor loss
            pi, log_pi, _ = self.agent.actor[agent_idx].get_action(observations[agent_idx])
            qf1_pi = self.agent.qf1[agent_idx](observations[agent_idx], pi)
            qf2_pi = self.agent.qf2[agent_idx](observations[agent_idx], pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            actor_loss = ((self.args.alpha * log_pi) - min_qf_pi).mean()

            # Optimize actor
            self.agent.actor_optimizer[agent_idx].zero_grad()
            actor_loss.backward()
            self.agent.actor_optimizer[agent_idx].step()
            
            # Update the target networks
            self.update_target_networks(agent_idx)
            
            # Log statistics
            self.log_stats(global_step, start_time, agent_idx, qf1_a_values, qf2_a_values, qf1_loss, qf2_loss, qf_loss, actor_loss)

    def train_vdn(self, global_step: int, start_time: float) -> None:
        """
        Train the VDN agent.
        
        Args:
            global_step (int): Current global step.
            start_time (float): Start time of the experiment.
        """
        qf1_a_values = torch.zeros(self.agent.num_agents, self.args.batch_size).to(self.device)
        qf2_a_values = torch.zeros(self.agent.num_agents, self.args.batch_size).to(self.device)
        next_q_values = torch.zeros(self.agent.num_agents, self.args.batch_size).to(self.device)

        for agent_idx in range(self.agent.num_agents):
            # Sampling different batches for each agent to avoid overfitting
            observations, actions, rewards, next_observations, dones = self.replay_buffer.sample(self.args.batch_size, to_gpu=self.args.cuda)
            
            # Calculate the target Q values
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = self.agent.actor[agent_idx].get_action(next_observations[agent_idx])
                qf1_next_target = self.agent.qf1_target[agent_idx](next_observations[agent_idx], next_state_actions)
                qf2_next_target = self.agent.qf2_target[agent_idx](next_observations[agent_idx], next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.args.alpha * next_state_log_pi
                next_q_values[agent_idx] = rewards[agent_idx] + (1 - dones[agent_idx]) * self.args.gamma * (min_qf_next_target).squeeze()
        
            qf1_a_values[agent_idx] = self.agent.qf1[agent_idx](observations[agent_idx], actions[agent_idx]).squeeze()
            qf2_a_values[agent_idx] = self.agent.qf2[agent_idx](observations[agent_idx], actions[agent_idx]).squeeze()
        
        # Calculate the joint Q values
        # As VDN just calculates the sum of the Q values, it doesn't need to calculate the target Q values
        joint_qf1_a_values = self.agent.vdn_mixer(qf1_a_values)
        joint_qf2_a_values = self.agent.vdn_mixer(qf2_a_values)
        joint_next_q_value = self.agent.vdn_mixer(next_q_values)
            
        qf1_loss = F.mse_loss(joint_qf1_a_values, joint_next_q_value)
        qf2_loss = F.mse_loss(joint_qf2_a_values, joint_next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # Optimize critic
        for agent_idx in range(self.agent.num_agents):
            self.agent.q_optimizer[agent_idx].zero_grad()
        
        qf_loss.backward()
        
        for agent_idx in range(self.agent.num_agents):
            self.agent.q_optimizer[agent_idx].step()
        
            # Calculate actor loss
            pi, log_pi, _ = self.agent.actor[agent_idx].get_action(observations[agent_idx])
            qf1_pi = self.agent.qf1[agent_idx](observations[agent_idx], pi)
            qf2_pi = self.agent.qf2[agent_idx](observations[agent_idx], pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            actor_loss = ((self.args.alpha * log_pi) - min_qf_pi).mean()

            # Optimize actor
            self.agent.actor_optimizer[agent_idx].zero_grad()
            actor_loss.backward()
            self.agent.actor_optimizer[agent_idx].step()
            
            # Update the target networks
            self.update_target_networks(agent_idx)
            
            # Log statistics
            self.log_stats(global_step, start_time, agent_idx, qf1_a_values, qf2_a_values, qf1_loss, qf2_loss, qf_loss, actor_loss)

    def sample_actions(self, obs: np.array, global_step: int) -> list[np.ndarray]:
        """
        Sample actions from the agent.
        
        If the global step is less than the learning starts, sample random actions.
        
        Args:
            obs (np.array): Observations from the environment.
            global_step (int): Current global step.
            
        Returns:
            list[np.ndarray]: Actions sampled from the agent.
        """
        if global_step < self.args.learning_starts:
            agent_actions = np.array([[self.env.action_space[0][agent_i].sample() for agent_i in range(self.agent.num_agents)]
                            for _ in range(self.env.n_envs)], dtype=object)
        with torch.no_grad():
            torch_obs = [torch.tensor(np.vstack(obs[:, i]), dtype=torch.float32).to(self.device)
                        for i in range(self.agent.num_agents)]
            agent_actions, _, _ = self.agent.get_action(torch_obs)
            agent_actions = [action.cpu().detach().numpy() for action in agent_actions]
            # rearrange actions to be per environment
            agent_actions = [[ac[i] for ac in agent_actions] for i in range(self.env.n_envs)]
        return agent_actions
    
    def step_and_push(self, obs: np.array, agent_actions: list[np.ndarray]) -> tuple[np.array, np.array]:
        """
        Step the environment and push the data to the replay buffer.
        
        Args:
            obs (np.array): Observations from the environment.
            agent_actions (list[np.ndarray]): Actions sampled from the agent.
            global_step (int): Current global step.
            
        Returns:
            tuple[np.array, np.array]: Next observations and terminations from the environment.
        """
        next_obs, rewards, terminations, truncations, infos = self.env.step(agent_actions)
        # rearrange actions to be per agent
        actions = [[ac[i] for ac in agent_actions] for i in range(self.agent.num_agents)]
        self.replay_buffer.push(obs, actions, rewards, next_obs, terminations)
        return next_obs, terminations
    
    def update_target_networks(self, agent_idx: int) -> None:
        """
        Update the target networks.
        
        Args:
            agent_idx (int): Index of the agent.
        """
        for param, target_param in zip(self.agent.qf1[agent_idx].parameters(), self.agent.qf1_target[agent_idx].parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
        for param, target_param in zip(self.agent.qf2[agent_idx].parameters(), self.agent.qf2_target[agent_idx].parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
    
    def set_seed(self, seed: int) -> None:
        """
        Set the seed for reproducibility of the experiment.
        
        Args:
            seed (int): Seed for the random number generator.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed) 
    