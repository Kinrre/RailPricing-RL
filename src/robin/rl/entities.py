"""Entities for the rl module."""

import numpy as np
import torch

from robin.kernel.entities import Kernel
from robin.supply.entities import Supply
from robin.rl.constants import ACTION_FACTOR, LOW_ACTION, HIGH_ACTION, NUMBER_ACTIONS, \
    START_ACTION, LOW_PRICE, HIGH_PRICE, CLIP_MAX, LOG_DIR

from abc import ABC, abstractmethod
from copy import deepcopy
from functools import cached_property, lru_cache
from gymnasium import ActionWrapper, Env, ObservationWrapper
from gymnasium import spaces
from gymnasium.spaces.utils import flatten_space, flatten, unflatten
from gymnasium.wrappers import FlattenObservation
from numpy.typing import NDArray
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tianshou.env import SubprocVectorEnv, VectorEnvNormObs
from tianshou.env.venvs import BaseVectorEnv
from tianshou.utils import RunningMeanStd
from tianshou.env.utils import gym_new_venv_step_type
from typing import Tuple, Union


class FlattenAction(ActionWrapper):
    """Action wrapper that flattens the action space."""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = flatten_space(self.env.action_space)
        
    def action(self, action):
        return unflatten(self.env.action_space, action)


class FlattenMultiAction(ActionWrapper):
    """Action wrapper that flattens the action space for multiple agents."""

    def __init__(self, env: Env):
        super().__init__(env)
        self.action_space = [flatten_space(space) for space in self.env.action_space]

    def action(self, action: list):
        return [unflatten(space, act) for space, act in zip(self.env.action_space, action)]


class FlattenObservation(FlattenObservation):
    pass


class FlattenMultiObservation(ObservationWrapper):
    """Observation wrapper that flattens the observation space for multiple agents."""

    def __init__(self, env: Env):
        super().__init__(env)
        self.observation_space = [flatten_space(space) for space in self.env.observation_space]

    def observation(self, observation: list):
        return [flatten(space, obs) for space, obs in zip(self.env.observation_space, observation)]


class HeterogeneousRunningMeanStd(RunningMeanStd):
    """
    Calculate the running mean and std of a data stream.
    
    NOTE: This class supports heterogeneous data types.
    """
    
    def norm(self, data_array: float | np.ndarray) -> float | np.ndarray:
        """
        Normalize the data array.
        
        Args:
            data_array (float | np.ndarray): Data array to normalize.
        
        Returns:
            float | np.ndarray: Normalized data array.
        """
        var = np.array([np.sqrt(var + self.eps) for var in self.var], dtype=object)
        data_array = (data_array - self.mean) / var
        if self.clip_max:
            data_array = np.reshape(
                np.array(
                    [np.clip(agent, -self.clip_max, self.clip_max) for data in data_array for agent in data],
                    dtype=object,
                ), data_array.shape
            )
        return data_array
    

class VectorEnvNormObsReward(VectorEnvNormObs):
    """
    Vector environment with normalized observations and rewards.

    Attributes:
        obs_rms (RunningMeanStd): Running mean/std for the observations.
        reward_rms (RunningMeanStd): Running mean/std for the rewards.
        update_reward_rms (bool): Whether to update the reward running mean/std.
    """

    def __init__(
        self,
        venv: BaseVectorEnv,
        update_obs_rms: bool = True,
        update_reward_rms: bool = True,
        clip_max: float = CLIP_MAX,
        is_heterogeneous: bool = False
    ) -> None:
        """
        Initialize the vector environment with normalized observations and rewards.

        Args:
            venv (BaseVectorEnv): Vector environment.
            update_obs_rms (bool): Whether to update the observation running mean/std.
            update_reward_rms (bool): Whether to update the reward running mean/std.
            clip_max (float): Maximum absolute value for the data array.
            is_heterogeneous (bool): Whether the data array is heterogeneous.
        """
        super().__init__(venv, update_obs_rms)
        self.obs_rms = HeterogeneousRunningMeanStd(clip_max=clip_max) if is_heterogeneous \
            else RunningMeanStd(clip_max=clip_max)
        self.update_reward_rms = update_reward_rms
        self.reward_rms = HeterogeneousRunningMeanStd(clip_max=clip_max) if is_heterogeneous \
            else RunningMeanStd(clip_max=clip_max)

    def step(
        self,
        action: Union[np.ndarray, torch.Tensor],
        id: Union[int, list[int], np.ndarray, None] = None,
    ) -> gym_new_venv_step_type:
        # Normalize observation
        step_results = super().step(action, id)
        # Normalize reward
        if self.reward_rms and self.update_reward_rms:
            self.reward_rms.update(step_results[1])
        return (step_results[0], self._norm_reward(step_results[1]), *step_results[2:])
    
    def _norm_reward(self, reward: float) -> np.ndarray:
        """Normalize the reward."""
        if self.reward_rms:
            return self.reward_rms.norm(reward)
        return reward

    def set_reward_rms(self, reward_rms: RunningMeanStd) -> None:
        """Set with given reward running mean/std."""
        self.reward_rms = reward_rms
    
    def get_reward_rms(self) -> RunningMeanStd:
        """Return reward running mean/std."""
        return self.reward_rms


class Stats:
    """
    Stats class to log data from the environment using a SummaryWriter.
    
    Attributes:
        agents (list[str]): Agents in the environment.
        num_agents (int): Number of agents in the environment.
        episode_length (int): Length of the episode.
        logger (SummaryWriter): A TensorboardX SummaryWriter instance for logging.
    """
    
    def __init__(self, agents: list[str], num_agents: int, episode_length: int, log_dir: str = LOG_DIR) -> None:
        """
        Initializes the Stats object to log data using SummaryWriter.
        
        Args:
            agents (list[str]): List of agents in the environment.
            num_agents (int): Number of agents in the environment.
            episode_length (int): Length of the episode.
            log_dir (str): The directory to save the logs.
        """
        self.agents = agents
        self.num_agents = num_agents
        self.episode_length = episode_length
        self.logger = SummaryWriter(log_dir)
        
    def log_agents_to_tensorboard(self, stats: list[dict], returns: np.ndarray[float], ep_i: int) -> None:
        """
        Log the agents data to Tensorboard for a specific episode.
        
        Args:
            stats (list[dict]): List of dictionaries containing stats at the end of an episode.
            returns (np.ndarray[float]): Returns of the environments.
            ep_i (int): The episode index.
        """
        # Log mean returns
        mean_returns = np.mean(returns, axis=0)
        self._log_agent_metric_to_tensorboard(dict(zip(self.agents, mean_returns)), 'mean_return', ep_i)
        
        # Log mean profit
        mean_profits = self._calculate_mean_agent_metric([info['agents']['profit'] for info in stats])
        self._log_agent_metric_to_tensorboard(mean_profits, 'mean_profit', ep_i)

        # Calculate mean profits
        profits = np.array([list(info['agents']['profit'].values()) for info in stats], dtype=np.float32)
        mean_profits = np.mean(profits, axis=0)

        # Log efficiency
        mean_efficiency = self._calculate_mean_efficiency(mean_profits)
        self.logger.add_scalar('agents/mean_efficiency', mean_efficiency, ep_i)

        # Log equality
        mean_equality = self._calculate_mean_equality(mean_profits)
        self.logger.add_scalar('agents/mean_equality', mean_equality, ep_i)

    def log_services_to_tensorboard(self, stats: list[dict], ep_i: int) -> None:
        """
        Logs the services data to Tensorboard for a specific episode.
        
        Args:
            stats (list[dict]): List of dictionaries containing stats at the end of an episode.
            ep_i (int): The episode index.
        """
        # Log mean total profit
        mean_total_profit = np.mean([info['services']['total_profit'] for info in stats])
        self.logger.add_scalar('services/mean_total_profit', mean_total_profit, ep_i)

        # Log mean prices and tickets sold for each service, market, and seat
        mean_prices = self._calculate_mean_service_metric([info['services']['prices'] for info in stats])
        self._log_service_metric_to_tensorboard(mean_prices, 'mean_last_prices', ep_i)
        mean_tickets_sold = self._calculate_mean_service_metric([info['services']['tickets_sold'] for info in stats])
        self._log_service_metric_to_tensorboard(mean_tickets_sold, 'mean_tickets_sold', ep_i)
        
    def log_passengers_to_tensorboard(self, stats: list[dict], ep_i: int) -> None:
        """
        Logs the passengers data to Tensorboard for a specific episode.
        
        Args:
            stats (list[dict]): List of dictionaries containing stats at the end of an episode.
            ep_i (int): The episode index.
        """
        # Log mean total passengers
        mean_total_passengers = np.mean([info['passengers']['total'] for info in stats])
        self.logger.add_scalar('passengers/mean_total_passengers', mean_total_passengers, ep_i)
        
        # Log mean total passengers travelling
        mean_passengers_travelling = np.mean([info['passengers']['travelling'] for info in stats])
        self.logger.add_scalar('passengers/mean_passengers_travelling', mean_passengers_travelling, ep_i)
        
        # Log mean total passengers not travelling
        mean_passengers_not_travelling = np.mean([info['passengers']['not_travelling'] for info in stats])
        self.logger.add_scalar('passengers/mean_passengers_not_travelling', mean_passengers_not_travelling, ep_i)
        
        # Log mean total percentage of passengers travelling
        mean_percentage_travelling = np.mean([info['passengers']['percentage_travelling'] for info in stats])
        self.logger.add_scalar('passengers/mean_percentage_travelling', mean_percentage_travelling, ep_i)
        
        # Log mean utility
        mean_utility = np.mean([info['passengers']['utility'] for info in stats])
        self.logger.add_scalar('passengers/mean_utility', mean_utility, ep_i)

    def to_tensorboard(self, stats: list[dict], returns: np.ndarray[float], ep_i: int) -> None:
        """
        Logs the stats to Tensorboard for a specific episode.
        
        Args:
            stats (list[dict]): List of dictionaries containing stats at the end of an episode.
            returns (np.ndarray[float]): Returns of the agents.
            ep_i (int): The episode index.
        """
        self.log_agents_to_tensorboard(stats, returns, ep_i)
        self.log_services_to_tensorboard(stats, ep_i)
        self.log_passengers_to_tensorboard(stats, ep_i)

    def _calculate_mean_agent_metric(self, agent_metric_list: list[dict[str, float]]) -> dict[str, float]:
        """
        Calculates the mean metric for each agent.
        
        Args:
            agent_metric_list (list[dict[str, float]]): List of dictionaries containing the metric for each agent.
        """
        aggregated_metric = {}

        # Aggregate metric for each agent
        for env in agent_metric_list:
            for agent, value in env.items():
                if agent not in aggregated_metric:
                    aggregated_metric[agent] = []
                aggregated_metric[agent].append(value)

        # Calculate the mean for each agent
        for agent, values in aggregated_metric.items():
            aggregated_metric[agent] = np.mean(values)

        return aggregated_metric

    def _calculate_mean_efficiency(self, profits: np.ndarray[float]) -> float:
        """
        Calculates the mean efficiency of the agents.

        Args:
            profits (np.ndarray[float]): Profits of the agents.

        Returns:
            float: Mean efficiency of the agents.
        """
        return np.sum(profits) / self.episode_length

    def _calculate_mean_equality(self, profits: np.ndarray[float]) -> float:
        """
        Calculates the mean equality of the agents.

        Args:
            profits (np.ndarray[float]): Profits of the agents.

        Returns:
            float: Mean equality of the agents.
        """
        pairwise_differences = np.sum(np.abs(profits[:, np.newaxis] - profits))
        normalization_factor = 2 * self.num_agents * np.sum(profits)
        return 1 - pairwise_differences / normalization_factor

    def _calculate_mean_service_metric(self, service_metric_list: list[dict[str, dict[str, dict[str, float]]]]) -> dict[str, dict[str, dict[str, float]]]:
        """
        Calculates the mean service metric for each service.
        
        Args:
            service_metric_list (list[dict[str, dict[str, dict[str, float]]]): List of dictionaries containing the
                service metric for each service, market and seat.
        """
        aggregated_metric = {}

        # Aggregate metric for each service, market, and seat
        for env in service_metric_list:
            for service, markets in env.items():
                if service not in aggregated_metric:
                    aggregated_metric[service] = {}
                for market, seats in markets.items():
                    if market not in aggregated_metric[service]:
                        aggregated_metric[service][market] = {}
                    for seat, value in seats.items():
                        if seat not in aggregated_metric[service][market]:
                            aggregated_metric[service][market][seat] = []
                        aggregated_metric[service][market][seat].append(value)

        # Calculate the mean for each seat
        for service, markets in aggregated_metric.items():
            for market, seats in markets.items():
                for seat, values in seats.items():
                    aggregated_metric[service][market][seat] = np.mean(values)

        return aggregated_metric
    
    def _log_agent_metric_to_tensorboard(self, metric: dict[str, float], metric_name: str, ep_i: int):
        """
        Logs the calculated mean metric for each agent to Tensorboard.
        
        Args:
            metric (dict[str, float]): Dictionary containing the calculated mean metric for each agent.
            metric_name (str): The name of the metric to log.
            ep_i (int): The episode index.
        """
        for agent, value in metric.items():
            self.logger.add_scalar(f'agents/{metric_name}/{agent}', value, ep_i)

    def _log_service_metric_to_tensorboard(self, metric: dict[str, dict[str, dict[str, float]]], metric_name: str, ep_i: int):
        """
        Logs the calculated mean tickets sold for each service, market, and seat to Tensorboard.
        
        Args:
            metric (dict[str, dict[str, dict[str, float]]]): Dictionary containing the calculated mean metric for each service, market, and seat.
            metric_name (str): The name of the metric to log.
            ep_i (int): The episode index.
        """
        for service, markets in metric.items():
            for market, seats in markets.items():
                for seat, value in seats.items():
                    self.logger.add_scalar(f'services/{metric_name}/{service}/{market}/{seat}', value, ep_i)


class StatsSubprocVectorEnv(SubprocVectorEnv):
    """
    Subprocess vectorized environment with stats logging.
    
    Attributes:
        stats (Stats): The Stats object to log data.
        episode_index (int): The episode index.
        n_envs (int): Number of environments.
        num_agents (int): Number of agents in the environments.
        episode_length (int): Length of the episode.
        returns (np.array[float]): Returns of the environments.
    """
    
    def __init__(self, log_dir: str = LOG_DIR, *args, **kwargs) -> None:
        """
        Initializes the StatsSubprocVectorEnv object with stats logging.
        
        Args:
            log_dir (str): The directory to save the logs.
        """
        super().__init__(*args, **kwargs)
        self.episode_index = 0
        self.n_envs = len(self.workers)
        self.agents = self.get_env_attr('agents')[0]
        self.num_agents = self.get_env_attr('num_agents')[0]
        self.episode_length = len(self.get_env_attr('kernel')[0].simulation_days)
        self.returns = np.zeros((self.n_envs, self.num_agents), dtype=np.float32)
        self.stats = Stats(self.agents, self.num_agents, self.episode_length, log_dir)
    
    def step(self, action: list, *args, **kwargs) -> Tuple[list, float, bool, bool, dict]:
        """
        Perform an action in the environment and log the stats.

        Args:
            action (list): Action to perform.
        
        Returns:
            Tuple[list, float, bool, bool, dict]: Observation, reward, termination, truncation and info of the environment.
        """
        obs, reward, terminated, truncated, info = super().step(action, *args, **kwargs)
        self.returns += reward
        if terminated.all():
            self.stats.to_tensorboard(info, self.returns, self.episode_index)
            self.returns = np.zeros((self.n_envs, self.num_agents), dtype=np.float32)
            self.episode_index += self.n_envs
        return obs, reward, terminated, truncated, info


class BaseRobinEnv(ABC):
    """
    Abstract class for the Robin simulator environment.

    Attributes:
        path_config_supply (Path): Path to the supply configuration file.
        path_config_demand (Path): Path to the demand configuration file.
        departure_time_hard_restriction (bool): Whether to apply a hard restriction to the departure time.
        kernel (Kernel): Kernel of the Robin simulator.
        action_factor (int): Factor to multiply the price action.
    """

    def __init__(
            self,
            path_config_supply: Path,
            path_config_demand: Path,
            departure_time_hard_restriction: bool = False,
            discrete_action_space: bool = False,
            action_factor: int = ACTION_FACTOR,
            seed: Union[int, None] = None
    ) -> None:
        """
        Initialize the environment.

        Args:
            path_config_supply (Path): Path to the supply configuration file.
            path_config_demand (Path): Path to the demand configuration file.
            departure_time_hard_restriction (bool): Whether to apply a hard restriction to the departure time.
            discrete_action_space (bool): Whether the action space is discrete or continuous.
            action_factor (int): Factor to multiply the price action (discrete). If the action space is continuous
                it is adjusted by multiplying the number of actions and dividing by half.
            seed (int, None): Seed for the random number generator.
        """
        self.path_config_supply = path_config_supply
        self.path_config_demand = path_config_demand
        self.departure_time_hard_restriction = departure_time_hard_restriction
        self.kernel = Kernel(self.path_config_supply, self.path_config_demand, seed)
        self.discrete_action_space = discrete_action_space
        self.action_factor = action_factor if discrete_action_space else action_factor * (NUMBER_ACTIONS / 2)

    @lru_cache(maxsize=None)
    def _get_element_idx_from_id(self, elements: tuple, id: str) -> int:
        """
        Get the index of an object element from its id attribute given a tuple.

        Args:
            elements (tuple): Tuple of elements.
            id (str): Id of the element.
        
        Returns:
            int: Index of the element.
        """
        return elements.index(next(filter(lambda x: x.id == id, elements)))

    def _get_obs(self, supply: Supply) -> list:
        """
        Get the observation of the environment.

        Args:
            supply (Supply): Supply of the environment.

        Returns:
            list: Observation of the environment.
        """
        obs = [
            {
                'tsp': self._get_element_idx_from_id(supply.tsps, service.tsp.id),
                'line': self._get_element_idx_from_id(supply.lines, service.line.id),
                'corridor': self._get_element_idx_from_id(supply.corridors, service.line.corridor.id),
                'time_slot': self._get_element_idx_from_id(supply.time_slots, service.time_slot.id),
                'rolling_stock': self._get_element_idx_from_id(supply.rolling_stocks, service.rolling_stock.id),
                'prices': [{
                    'origin': self._get_element_idx_from_id(supply.stations, origin),
                    'destination': self._get_element_idx_from_id(supply.stations, destination),
                    'seats': [{
                        'seat_type': self._get_element_idx_from_id(supply.seats, seat.id),
                        'price': price
                    } for seat, price in seats.items()]
                } for (origin, destination), seats in service.prices.items()],
                'tickets_sold': [{
                    'origin': self._get_element_idx_from_id(supply.stations, origin),
                    'destination': self._get_element_idx_from_id(supply.stations, destination),
                    'seats': [{
                        'seat_type': self._get_element_idx_from_id(supply.seats, seat.id),
                        'count': count
                    } for seat, count in seats.items()]
                } for (origin, destination), seats in service.tickets_sold_pair_seats.items()]
            }
            for service in supply.services
        ]
        return obs

    def _get_info(self) -> dict:
        """
        Get the info of the environment.

        Returns:
            dict: Info of the environment.
        """
        profit = [service.total_profit for service in self.kernel.supply.services]
        agents_profit = {tsp.name: sum(service.total_profit for service in self.kernel.supply.services if service.tsp.id == tsp.id) for tsp in self.kernel.supply.tsps}
        total_passengers = len(self.kernel.passengers)
        traveling_passengers = len([passenger for passenger in self.kernel.passengers if passenger.journey])
        info = {
            'agents': {
                'profit': agents_profit
            },
            'services': {
                'total_profit': sum(profit),
                'profit': profit,
                'prices': {
                    service.id: {
                        '_'.join(market): {
                            seat.name: price for seat, price in seats.items()
                        } for market, seats in service.prices.items()
                    } for service in self.kernel.supply.services
                },
                'tickets_sold': {
                    service.id: {
                        '_'.join(market): {
                            seat.name: count for seat, count in seats.items()
                        } for market, seats in service.tickets_sold_pair_seats.items()
                    } for service in self.kernel.supply.services
                }
            },
            'passengers': {
                'total': total_passengers,
                'travelling': traveling_passengers,
                'not_travelling': total_passengers - traveling_passengers,
                'percentage_travelling': traveling_passengers / total_passengers * 100,
                'utility': np.mean([passenger.utility for passenger in self.kernel.passengers])
            }
        }
        return info

    @abstractmethod
    def _get_reward(self) -> float:
        """
        Get the reward of the environment.

        The total profit of the services of a day is the reward.

        Returns:
            float: Reward of the environment.
        """
        # NOTE: Don't forget about if we want to add negative rewards as costs
        # We can also promote the some specific services, such as direct services by adding a weight to the reward
        raise NotImplementedError

    def _get_terminated(self) -> bool:
        """
        Get the termination of the environment.

        The environment is terminated when the simulation is finished.

        Returns:
            bool: Termination of the environment.
        """
        return self.kernel.is_simulation_finished

    def reset(self, seed: Union[int, None] = None, options: dict = None) -> Tuple[list, dict]:
        """
        Reset the environment.

        It only sets the seed and re-creates the kernel. Child classes should implement the rest of the reset.

        Args:
            seed (int, None): Seed for the random number generator.
            options (dict, None): Options for the reset.
        
        Returns:
            Tuple[list, dict]: Observation and info of the environment.
        """
        super().reset(seed=seed)
        # NOTE: If the config files are changed during the simulation, it will load the new ones, beware!
        self.kernel = Kernel(self.path_config_supply, self.path_config_demand, seed)

    def _update_prices(self, action: list, supply: Supply) -> None:
        """
        Update the prices of the services in the kernel supply by multiplying the price action by a factor.

        Args:
            action (list): Action to perform.
            supply (Supply): Supply of the environment.
        """
        for service, action_service in zip(supply.services, action):
            for ((origin, destination), seats), price in zip(service.prices.items(), action_service['prices']):
                for seat_type, seat_price in zip(seats, price['seats']):
                    price_modification = seat_price['price'] * (self.action_factor / 100)
                    service.prices[(origin, destination)][seat_type] *= (1 + price_modification)
                    # Clip the price to its range, so, it is not possible to have negative prices
                    service.prices[(origin, destination)][seat_type] = \
                        np.clip(service.prices[(origin, destination)][seat_type], LOW_PRICE, HIGH_PRICE)

    def _step(self, action: list, supply: Union[Supply, list[Supply]]) -> None:
        """
        Private method to perform an action in the environment.

        Args:
            action (list): Action to perform.
            supply (Union[Supply, list[Supply]]): Supply or supplies of the environment.
        """
        if isinstance(supply, list):
            for supply_action, supply_ in zip(action, supply):
                self._update_prices(action=supply_action, supply=supply_)
        else:
            self._update_prices(action=action, supply=supply)
        self.kernel.simulate_a_day(departure_time_hard_restriction=self.departure_time_hard_restriction)

    @abstractmethod
    def step(self, action: list) -> Tuple[list, float, bool, bool, dict]:
        """
        Perform an action in the environment.

        Args:
            action (list): Action to perform.
        
        Returns:
            Tuple[list, float, bool, bool, dict]: Observation, reward, termination, truncation and info of the environment.
        """
        raise NotImplementedError

    def seed(self, seed: int) -> None:
        """
        Set seed for the random number generator.

        Args:
            seed (int): Seed for the random number generator.
        """
        self.kernel.set_seed(seed)

    @lru_cache(maxsize=None)
    def observation_space(self, supply: Supply) -> spaces.Space:
        """
        Observation space of the environment.

        Args:
            supply (Supply): Supply of the environment.

        Returns:
            spaces.Space: Observation space of the environment.
        """
        observation_space = spaces.Tuple([
            spaces.Dict({
                # service already departed for an action mask?
                # date time details? day of the week?
                # capacity of the rolling stock?
                'tsp': spaces.Box(low=(idx := self._get_element_idx_from_id(supply.tsps, service.tsp.id)), high=idx, shape=(), dtype=np.int32),
                'line': spaces.Box(low=(idx := self._get_element_idx_from_id(supply.lines, service.line.id)), high=idx, shape=(), dtype=np.int32),
                'corridor': spaces.Box(low=(idx := self._get_element_idx_from_id(supply.corridors, service.line.corridor.id)), high=idx, shape=(), dtype=np.int32),
                'time_slot': spaces.Box(low=(idx := self._get_element_idx_from_id(supply.time_slots, service.time_slot.id)), high=idx, shape=(), dtype=np.int32),
                'rolling_stock': spaces.Box(low=(idx := self._get_element_idx_from_id(supply.rolling_stocks, service.rolling_stock.id)), high=idx, shape=(), dtype=np.int32),
                'prices': spaces.Tuple([
                    spaces.Dict({
                        'origin': spaces.Box(low=(idx := self._get_element_idx_from_id(supply.stations, origin)), high=idx, shape=(), dtype=np.int32),
                        'destination': spaces.Box(low=(idx := self._get_element_idx_from_id(supply.stations, destination)), high=idx, shape=(), dtype=np.int32),
                        'seats': spaces.Tuple([
                            spaces.Dict({
                                'seat_type': spaces.Box(low=(idx := self._get_element_idx_from_id(supply.seats, seat.id)), high=idx, shape=(), dtype=np.int32),
                                'price': spaces.Box(low=LOW_PRICE, high=HIGH_PRICE, shape=(), dtype=np.float32)
                            }) for seat, _ in seats.items()
                        ])
                    }) for (origin, destination), seats in service.prices.items()
                ]),
                'tickets_sold': spaces.Tuple([
                    spaces.Dict({
                        'origin': spaces.Box(low=(idx := self._get_element_idx_from_id(supply.stations, origin)), high=idx, shape=(), dtype=np.int32),
                        'destination': spaces.Box(low=(idx := self._get_element_idx_from_id(supply.stations, destination)), high=idx, shape=(), dtype=np.int32),
                        'seats': spaces.Tuple([
                            spaces.Dict({
                                'seat_type': spaces.Box(low=(idx := self._get_element_idx_from_id(supply.seats, seat.id)), high=idx, shape=(), dtype=np.int32),
                                'count': spaces.Box(low=0, high=service.rolling_stock.total_capacity, shape=(), dtype=np.int32)
                            }) for seat, _ in seats.items()
                        ])
                    }) for (origin, destination), seats in service.tickets_sold_pair_seats.items()
                ]),
            }) for service in supply.services
        ])
        return observation_space

    @lru_cache(maxsize=None)
    def action_space(self, supply: Supply) -> spaces.Space:
        """
        Action space of the environment.

        Args:
            supply (Supply): Supply of the environment.

        Returns:
            spaces.Space: Action space of the environment.
        """
        action_space = spaces.Tuple([
            spaces.Dict({
                'prices': spaces.Tuple([
                    spaces.Dict({
                        'seats': spaces.Tuple([
                            spaces.Dict({
                                'price': spaces.Discrete(n=NUMBER_ACTIONS, start=START_ACTION) if self.discrete_action_space \
                                    else spaces.Box(low=LOW_ACTION, high=HIGH_ACTION, shape=(), dtype=np.float32)
                            }) for _ in seats
                        ])
                    }) for _, seats in service.prices.items()
                ])
            }) for service in supply.services
        ])
        return action_space


class RobinSingleAgentEnv(BaseRobinEnv, Env):
    """
    Reinforcement learning single-agent environment for the Robin simulator.
    
    Attributes:
        agents (list[str]): Agents in the environment.
        num_agents (int): Number of agents in the environment.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the single-agent environment.
        """
        super().__init__(*args, **kwargs)
        self.agents = [tsp.name for tsp in self.kernel.supply.tsps]
        self.num_agents = 1
        self._last_total_profit = 0

    def _get_reward(self) -> float:
        """
        Get the reward of the environment.

        The total profit of the services of a day is the reward.

        Returns:
            float: Reward of the environment.
        """
        total_profit = sum(service.total_profit for service in self.kernel.supply.services)
        reward = total_profit - self._last_total_profit
        self._last_total_profit = total_profit
        return reward

    def step(self, action: list) -> Tuple[list, float, bool, bool, dict]:
        """
        Perform an action in the environment.

        Args:
            action (list): Action to perform.
        
        Returns:
            Tuple[list, float, bool, bool, dict]: Observation, reward, termination, truncation and info of the environment.
        """
        self._step(action=action, supply=self.kernel.supply)
        obs = self._get_obs(supply=self.kernel.supply)
        reward = self._get_reward()
        terminated = self._get_terminated()
        truncated = False
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Union[int, None] = None, options: dict = None) -> Tuple[list, dict]:
        """
        Reset the environment.

        Args:
            seed (int, None): Seed for the random number generator.
            options (dict, None): Options for the reset.
        
        Returns:
            Tuple[list, dict]: Observation and info of the environment.
        """
        super().reset(seed=seed, options=options)
        self._last_total_profit = 0
        obs = self._get_obs(supply=self.kernel.supply)
        info = self._get_info()
        return obs, info

    @cached_property
    def observation_space(self) -> spaces.Space:
        """
        Observation space of the environment.

        Returns:
            spaces.Space: Observation space of the environment.
        """
        return super().observation_space(supply=self.kernel.supply)

    @cached_property
    def action_space(self) -> spaces.Space:
        """
        Action space of the environment.

        Returns:
            spaces.Space: Action space of the environment.
        """
        return super().action_space(supply=self.kernel.supply)


class RobinMultiAgentEnv(BaseRobinEnv, Env):
    """
    Reinforcement learning multi-agent environment for the Robin simulator.

    Attributes:
        agents (list[str]): Agents in the environment.
        num_agents (int): Number of agents in the environment.
        supplies (list[Supply]): Supplies of the agents.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the multi-agent environment.
        """
        super().__init__(*args, **kwargs)
        self.agents = [tsp.name for tsp in self.kernel.supply.tsps]
        self.num_agents = len(self.agents)
        self.supplies = [self.kernel.filter_supply_by_tsp(tsp.id) for tsp in self.kernel.supply.tsps]
        self._last_total_profit = [0 for _ in self.agents]

    def _get_obs(self) -> list:
        """
        Get the observation of the environment.

        Returns:
            list: Observation of the environment.
        """
        observation = []
        full_observation = super()._get_obs(supply=self.kernel.supply)
        for agent in self.agents:
            obs = deepcopy(full_observation)
            for service in obs:
                if service['tsp'] != self.agents.index(agent):
                    service.pop('tickets_sold')
            observation.append(obs)
        return np.array(observation, dtype=object)

    def _get_reward(self, agent_idx: int, supply: Supply) -> float:
        """
        Get the reward of the environment.

        The total profit of the services of a day is the reward.

        Args:
            agent_idx (int): Index of an agent.
            supply (Supply): Supply of an agent.
        
        Returns:
            float: Reward of the environment.
        """
        total_profit = sum(service.total_profit for service in supply.services)
        reward = total_profit - self._last_total_profit[agent_idx]
        self._last_total_profit[agent_idx] = total_profit
        return reward

    def step(self, action: list) -> Tuple[NDArray, NDArray[np.float32], NDArray[np.bool_], NDArray[np.bool_], dict]:
        """
        Perform an action in the environment.

        Args:
            action (list): Action to perform.
        
        Returns:
            Tuple[list, float, bool, bool, dict]: Observation, reward, termination, truncation and info of the environment.
        """
        self._step(action=action, supply=self.supplies)
        obs = self._get_obs()
        reward = np.array([self._get_reward(agent_idx, supply) for agent_idx, supply in enumerate(self.supplies)])
        _terminated = self._get_terminated() # The terminated condition is the same for all agents
        terminated = np.array([_terminated for _ in self.supplies])
        truncated = np.array([False for _ in self.supplies])
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Union[int, None] = None, options: dict = None) -> Tuple[dict, dict]:
        """
        Reset the environment.

        Args:
            seed (int, None): Seed for the random number generator.
            options (dict, None): Options for the reset.

        Returns:
            Tuple[dict, dict]: Observation and info of the environment.
        """
        super().reset(seed=seed, options=options)
        # It is necessary to re-create the supplies with the new services references as the Kernel object is re-created
        self.supplies = [self.kernel.filter_supply_by_tsp(tsp.id) for tsp in self.kernel.supply.tsps]
        self._last_total_profit = [0 for _ in self.agents]
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    @cached_property
    def observation_space(self) -> spaces.Space:
        """
        Observation space of the environment.

        Each agent has the their own observation from the full observation space,
        except the tickets_sold key is removed if the tsp key is not from the agent.

        Returns:
            spaces.Space: Observation space of each agent in the environment.
        """
        observation_spaces = []
        full_observation_space = super().observation_space(supply=self.kernel.supply)
        for agent in self.agents:
            # As the observation space is a Tuple space, it is necessary to convert it to a list to modify it
            observation_space = list(deepcopy(full_observation_space))
            for i, service in enumerate(observation_space):
                if service['tsp'].low != self.agents.index(agent):
                    # It is not possible to directly remove the key from the Dict space
                    service = spaces.Dict({key: value for key, value in service.items() if key != 'tickets_sold'})
                    observation_space[i] = service
            observation_spaces.append(spaces.Tuple(observation_space))
        return spaces.Tuple(observation_spaces)

    @cached_property
    def action_space(self) -> spaces.Space:
        """
        Action space of the environment.

        Returns:
            spaces.Space: Action space of each agent in the environment.
        """
        # List comprehension can't be used with super() method
        action_spaces = []
        for supply in self.supplies:
            action_spaces.append(super().action_space(supply=supply))
        return spaces.Tuple(action_spaces)


class RobinMultiAgentCoopEnv(RobinMultiAgentEnv):
    
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the multi-agent cooperative environment.
        """
        super().__init__(*args, **kwargs)
        
    def _get_reward(self, agent_idx: int, supply: Supply) -> float:
        """
        Get the reward of the environment.

        The total profit of the services of a day is the reward.

        Args:
            agent_idx (int): Index of an agent.
            supply (Supply): Supply of an agent.
        
        Returns:
            float: Reward of the environment.
        """
        total_profit = sum(service.total_profit for service in self.kernel.supply.services)
        reward = total_profit - self._last_total_profit[agent_idx]
        self._last_total_profit[agent_idx] = total_profit
        return reward


class RobinEnvFactory:

    @staticmethod
    def create(
            path_config_supply: Path,
            path_config_demand: Path,
            multi_agent: bool = False,
            cooperative: bool = False,
            departure_time_hard_restriction: bool = False,
            discrete_action_space: bool = False,
            action_factor: int = ACTION_FACTOR,
            seed: Union[int, None] = None
    ) -> BaseRobinEnv:
        """
        Create a Robin environment.

        Args:
            path_config_supply (Path): Path to the supply configuration file.
            path_config_demand (Path): Path to the demand configuration file.
            multi_agent (bool): Whether to create a multi-agent environment.
            cooperative (bool): Whether to create a cooperative multi-agent environment.
            departure_time_hard_restriction (bool): Whether to apply a hard restriction to the departure time.
            discrete_action_space (bool): Whether the action space is discrete or continuous.
            action_factor (int): Factor to multiply the price action.
            seed (int, None): Seed for the random number generator.
        
        Returns:
            RobinEnv: Robin environment.
        """
        if multi_agent:
            if cooperative:
                env = RobinMultiAgentCoopEnv(
                    path_config_supply=path_config_supply,
                    path_config_demand=path_config_demand,
                    departure_time_hard_restriction=departure_time_hard_restriction,
                    discrete_action_space=discrete_action_space,
                    action_factor=action_factor,
                    seed=seed
                )
            else:
                env = RobinMultiAgentEnv(
                    path_config_supply=path_config_supply,
                    path_config_demand=path_config_demand,
                    departure_time_hard_restriction=departure_time_hard_restriction,
                    discrete_action_space=discrete_action_space,
                    action_factor=action_factor,
                    seed=seed
                )
            env = FlattenMultiObservation(env)
            env = FlattenMultiAction(env)
        else:
            env = RobinSingleAgentEnv(
                path_config_supply=path_config_supply,
                path_config_demand=path_config_demand,
                departure_time_hard_restriction=departure_time_hard_restriction,
                discrete_action_space=discrete_action_space,
                action_factor=action_factor,
                seed=seed
            )
            env = FlattenObservation(env)
            env = FlattenAction(env)
        return env
