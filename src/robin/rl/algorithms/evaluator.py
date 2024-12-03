"""Evaluator module for evaluating the performance of a policy."""

import numpy as np
import torch
import tyro

from dataclasses import dataclass
from typing import Literal, Union

from robin.rl.algorithms.constants import EVALUATOR_SEED_RANK_MULTIPLIER
from robin.rl.algorithms.iql_sac import IQLSAC
from robin.rl.algorithms.utils import create_env
from robin.rl.algorithms.vdn import VDN


@dataclass
class EvaluatorArgs:
    algorithm: Literal['iql_sac', 'vdn'] = 'iql_sac'
    """the algorithm to use"""
    input_dir: str = '<output_dir>/<algorithm>/<seed>/<date>'
    """the input directory to load the model from"""
    total_timesteps: int = 50_000
    """number of timesteps to train the agent"""
    cuda: bool = True
    """whether to use cuda"""
    seed: int = 0
    """seed of the experiment"""
    n_workers: int = 16
    """number of workers for evaluation (it should be the number of cores)"""


class Evaluator:
    """
    Evaluator class for evaluating the performance of a policy.

    Attributes:
        args (EvaluatorArgs): Arguments for the evaluator.
        device (torch.device): Device to run the evaluator on.
        model_path (str): Path to the model to load.
        evaluation_path (str): Path to save the evaluation results.
        env (StatsSubprocVectorEnv): The environment to evaluate the agent on.
        agent (Union[IQLSAC, VDN]): The agent to evaluate.
        episode_length (int): The length of the episode.
        n_episodes (int): The number of episodes to evaluate.
        policy_distribution (np.array): The distribution of actions taken by the agent.
    """

    def __init__(self) -> None:
        """
        Initialize the evaluator.
        
        It uses the arguments from the command line to initialize the evaluator,
        please refer to the EvaluatorArgs class.
        """
        self.args = tyro.cli(EvaluatorArgs)
        self.device = torch.device('cuda' if self.args.cuda and torch.cuda.is_available() else 'cpu')
        self.model_path = f'{self.args.input_dir}/model.pt'
        self.evaluation_path = f'{self.args.input_dir}/evaluation'
        self.env = create_env(
            supply_config=f'{self.args.input_dir}/supply_config.yml',
            demand_config=f'{self.args.input_dir}/demand_config.yml',
            algorithm=self.args.algorithm,
            seed=self.args.seed,
            seed_rank_multiplier=EVALUATOR_SEED_RANK_MULTIPLIER,
            n_workers=self.args.n_workers,
            run_name=self.evaluation_path
        )
        self.agent = self.load_model(self.args.algorithm, self.model_path)
        self.agent.eval()
        self.episode_length = len(self.env.get_env_attr('kernel')[0].simulation_days)
        # Initialize the policy distribution, shape: (num_agents, n_actions, n_episodes, episode_length)
        self.n_episodes = self.args.total_timesteps // self.episode_length
        self.policy_distribution = np.array(
            [np.zeros((acsp.shape[0], self.n_episodes, self.episode_length), dtype=np.float32)
             for acsp in self.env.action_space[0]],
            dtype=object
        )

    def evaluate(self) -> None:
        """
        Evaluate the policy using the given model.
        """
        obs, _ = self.env.reset()

        for global_step in range(0, self.args.total_timesteps, self.env.n_envs):
            # Sample actions from the agent and step the environment
            actions = self.sample_actions(obs)
            self.update_policy_distribution(actions, global_step)
            next_obs, _, terminations, _, _ = self.env.step(actions)

            # Update observations
            obs = next_obs
            if terminations.all():
                obs, _ = self.env.reset()
                
        # Save the policy distribution
        np.save(f'{self.evaluation_path}/policy_distribution.npy', self.policy_distribution)

    def sample_actions(self, obs: np.ndarray) -> list[np.ndarray]:
        """
        Sample actions from the agent.
        
        If the global step is less than the learning starts, sample random actions.
        
        Args:
            obs (np.array): Observations from the environment.
            
        Returns:
            actions (list[np.ndarray]): Actions sampled from the agent for each environment.
        """
        torch_obs = [torch.tensor(np.vstack(obs[:, i]), dtype=torch.float32).to(self.device)
                    for i in range(self.agent.num_agents)]
        agent_actions, _, _ = self.agent.get_action(torch_obs)
        agent_actions = [action.cpu().detach().numpy() for action in agent_actions]
        # rearrange actions to be per environment
        actions = [[ac[i] for ac in agent_actions] for i in range(self.env.n_envs)]
        return actions

    def load_model(self, algorithm: str, model_path: str) -> Union[IQLSAC | VDN]:
        """
        Load the model from the given path.

        Args:
            algorithm (str): The algorithm to use.
            model_path (str): Path to the model to load.

        Returns:
            Union[IQLSAC, VDN]: The loaded model.
        """
        if algorithm == 'iql_sac':
            return IQLSAC.load_model(model_path, self.env, self.device)
        elif algorithm == 'vdn':
            return VDN.load_model(model_path, self.env, self.device)
        else:
            raise Exception(f'Algorithm {algorithm} not supported.')

    def update_policy_distribution(self, actions: list[np.ndarray], global_step: int) -> None:
        """
        Update the policy distribution with the agent actions.

        Args:
            actions (list[np.ndarray]): Actions sampled from the agent.
            global_step (int): The global step of the training.
        """
        # rearrange actions to be per agent
        agents_actions = [[ac[i] for ac in actions] for i in range(self.agent.num_agents)]
        episode = (global_step // self.env.n_envs // self.episode_length) * self.env.n_envs
        timestep = global_step % self.episode_length
        for agent_id, agent_actions in enumerate(agents_actions):
            stacked_actions = np.vstack(agent_actions).transpose()
            self.policy_distribution[agent_id][:, episode:episode+self.env.n_envs, timestep] = \
                stacked_actions
