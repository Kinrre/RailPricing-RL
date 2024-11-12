"""Evaluator module for evaluating the performance of a policy."""

import numpy as np
import torch
import tyro

from dataclasses import dataclass
from typing import Literal, Union

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
        env (StatsSubprocVectorEnv): The environment to evaluate the agent on.
        agent (Union[IQLSAC, VDN]): The agent to evaluate.
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
        self.env = create_env(
            supply_config=f'{self.args.input_dir}/supply_config.yml',
            demand_config=f'{self.args.input_dir}/demand_config.yml',
            algorithm=self.args.algorithm,
            seed=self.args.seed,
            n_workers=self.args.n_workers,
            run_name=self.args.input_dir
        )
        self.agent = self.load_model(self.args.algorithm, self.model_path)

    def evaluate(self) -> None:
        """
        Evaluate the policy using the given model.
        """
        obs, _ = self.env.reset()

        for global_step in range(0, self.args.total_timesteps, self.env.n_envs):
            # Sample actions from the agent and step the environment
            agent_actions = self.sample_actions(obs)
            next_obs, rewards, terminations, _, _ = self.env.step(agent_actions)

            # Update observations
            obs = next_obs
            if terminations.all():
                obs, _ = self.env.reset()
            
            print(rewards)

    def sample_actions(self, obs: np.ndarray) -> list[np.ndarray]:
        """
        Sample actions from the agent.
        
        If the global step is less than the learning starts, sample random actions.
        
        Args:
            obs (np.array): Observations from the environment.
            
        Returns:
            agent_actions (list[np.ndarray]): Actions sampled from the agent.
        """
        torch_obs = [torch.tensor(np.vstack(obs[:, i]), dtype=torch.float32).to(self.device)
                    for i in range(self.agent.num_agents)]
        agent_actions, _, _ = self.agent.get_action(torch_obs)
        agent_actions = [action.cpu().detach().numpy() for action in agent_actions]
        # rearrange actions to be per environment
        agent_actions = [[ac[i] for ac in agent_actions] for i in range(self.env.n_envs)]
        return agent_actions

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
