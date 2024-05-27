"""Entities for the rl module."""

import numpy as np
import torch

from src.robin.kernel.entities import Kernel
from src.robin.supply.entities import Supply
from src.robin.rl.constants import ACTION_FACTOR, LOW_ACTION, HIGH_ACTION, LOW_PRICE, HIGH_PRICE

from abc import ABC, abstractmethod
from functools import cached_property, lru_cache
from gymnasium import ActionWrapper, Env
from gymnasium import spaces
from gymnasium.spaces.utils import flatten_space, flatten, unflatten
from gymnasium.wrappers import FlattenObservation
from pathlib import Path
from tianshou.env import VectorEnvNormObs
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

    def reverse_action(self, action):
        # NOTE: Not needed
        return flatten(self.env.action_space, action)


class FlattenObservation(FlattenObservation):
    pass


class VectorEnvNormObsReward(VectorEnvNormObs):
    """
    Vector environment with normalized observations and rewards.

    Attributes:
        reward_rms (RunningMeanStd): Running mean/std for the rewards.
        update_reward_rms (bool): Whether to update the reward running mean/std.
    """

    def __init__(self, venv: BaseVectorEnv, update_obs_rms: bool = True, update_reward_rms: bool = True) -> None:
        """
        Initialize the vector environment with normalized observations and rewards.

        Args:
            venv (BaseVectorEnv): Vector environment.
            update_obs_rms (bool): Whether to update the observation running mean/std.
            update_reward_rms (bool): Whether to update the reward running mean/std.
        """
        super().__init__(venv, update_obs_rms)
        self.update_reward_rms = update_reward_rms
        self.reward_rms = RunningMeanStd()

    def step(
        self,
        action: np.ndarray | torch.Tensor,
        id: int | list[int] | np.ndarray | None = None,
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


class RobinEnv(ABC):
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
            action_factor: int = ACTION_FACTOR,
            seed: Union[int, None] = None
    ) -> None:
        """
        Initialize the environment.

        Args:
            path_config_supply (Path): Path to the supply configuration file.
            path_config_demand (Path): Path to the demand configuration file.
            departure_time_hard_restriction (bool): Whether to apply a hard restriction to the departure time.
            action_factor (int): Factor to multiply the price action.
            seed (int, None): Seed for the random number generator.
        """
        self.path_config_supply = path_config_supply
        self.path_config_demand = path_config_demand
        self.departure_time_hard_restriction = departure_time_hard_restriction
        self.kernel = Kernel(self.path_config_supply, self.path_config_demand, seed)
        self.action_factor = action_factor
        self.seed(seed)

    @lru_cache(maxsize=None)
    def _get_element_idx_from_id(self, elements: tuple, id: str) -> int:
        """
        Get the index of an element from its id given a tuple.

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

    @abstractmethod
    def _get_info(self) -> dict:
        """
        Get the info of the environment.

        Returns:
            dict: Info of the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_reward(self) -> float:
        """
        Get the reward of the environment.

        The total profit of the services of a day is the reward.

        Returns:
            float: Reward of the environment.
        """
        raise NotImplementedError

    def _get_terminated(self) -> bool:
        """
        Get the termination of the environment.

        The environment is terminated when the simulation is finished.

        Returns:
            bool: Termination of the environment.
        """
        return self.kernel.is_simulation_finished

    @abstractmethod
    def reset(self, seed: Union[int, None] = None, options: dict = None) -> Tuple[list, dict]:
        """
        Reset the environment.

        Args:
            seed (int, None): Seed for the random number generator.
            options (dict, None): Options for the reset.
        
        Returns:
            Tuple[list, dict]: Observation and info of the environment.
        """
        raise NotImplementedError

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
                    price_modification = seat_price['price'] * self.action_factor
                    service.prices[(origin, destination)][seat_type] += price_modification
                    # Clip the price to its range, so, it is not possible to have negative prices
                    service.prices[(origin, destination)][seat_type] = \
                        np.clip(service.prices[(origin, destination)][seat_type], LOW_PRICE, HIGH_PRICE)

    def _step(self, action: list, supply: Supply) -> None:
        """
        Private method to perform an action in the environment.

        Args:
            action (list): Action to perform.
            supply (Supply): Supply of the environment.
        """
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
                'line': spaces.Box(low=(idx := self._get_element_idx_from_id(supply.lines, service.line.id)), high=idx, shape=(), dtype=np.int16),
                'corridor': spaces.Box(low=(idx := self._get_element_idx_from_id(supply.corridors, service.line.corridor.id)), high=idx, shape=(), dtype=np.int16),
                'time_slot': spaces.Box(low=(idx := self._get_element_idx_from_id(supply.time_slots, service.time_slot.id)), high=idx, shape=(), dtype=np.int16),
                'rolling_stock': spaces.Box(low=(idx := self._get_element_idx_from_id(supply.rolling_stocks, service.rolling_stock.id)), high=idx, shape=(), dtype=np.int16),
                'prices': spaces.Tuple([
                    spaces.Dict({
                        'origin': spaces.Box(low=(idx := self._get_element_idx_from_id(supply.stations, origin)), high=idx, shape=(), dtype=np.int16),
                        'destination': spaces.Box(low=(idx := self._get_element_idx_from_id(supply.stations, destination)), high=idx, shape=(), dtype=np.int16),
                        'seats': spaces.Tuple([
                            spaces.Dict({
                                'seat_type': spaces.Box(low=(idx := self._get_element_idx_from_id(supply.seats, seat.id)), high=idx, shape=(), dtype=np.int16),
                                'price': spaces.Box(low=LOW_PRICE, high=HIGH_PRICE, shape=(), dtype=np.float16)
                            }) for seat, _ in seats.items()
                        ])
                    }) for (origin, destination), seats in service.prices.items()
                ]),
                'tickets_sold': spaces.Tuple([
                    spaces.Dict({
                        'origin': spaces.Box(low=(idx := self._get_element_idx_from_id(supply.stations, origin)), high=idx, shape=(), dtype=np.int16),
                        'destination': spaces.Box(low=(idx := self._get_element_idx_from_id(supply.stations, destination)), high=idx, shape=(), dtype=np.int16),
                        'seats': spaces.Tuple([
                            spaces.Dict({
                                'seat_type': spaces.Box(low=(idx := self._get_element_idx_from_id(supply.seats, seat.id)), high=idx, shape=(), dtype=np.int16),
                                'count': spaces.Box(low=0, high=service.rolling_stock.total_capacity, shape=(), dtype=np.int16)
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
                                'price': spaces.Box(low=LOW_ACTION, high=HIGH_ACTION, shape=(), dtype=np.float16)
                            }) for _ in seats
                        ])
                    }) for _, seats in service.prices.items()
                ])
            }) for service in supply.services
        ])
        return action_space


class RobinSingleAgentEnv(Env, RobinEnv):
    """
    Reinforcement learning single-agent environment for the Robin simulator.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the single-agent environment.
        """
        super().__init__(*args, **kwargs)
        self._last_total_profit = 0

    def _get_info(self) -> dict:
        """
        Get the info of the environment.

        No info is provided for the moment.

        Returns:
            dict: Info of the environment.
        """
        return {}

    def _get_reward(self) -> float:
        """
        Get the reward of the environment.

        The total profit of the services of a day is the reward.

        Returns:
            float: Reward of the environment.
        """
        # NOTE: Think about how we can feed the reward in different components to the agent
        # as we have multiple services, markets and seats, this information can be useful
        # probably we need to normalize the reward by the number of services and add negative rewards as costs
        # We can also promote the some specific services, such as direct services by adding a weight to the reward
        reward = sum(service.total_profit for service in self.kernel.supply.services)
        reward -= self._last_total_profit
        self._last_total_profit = sum(service.total_profit for service in self.kernel.supply.services)
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
        super().reset(seed=seed)
        # NOTE: If the config files are changed during the simulation, it will load the new ones, beware!
        self.kernel = Kernel(self.path_config_supply, self.path_config_demand, seed)
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


class RobinEnvFactory:

    @staticmethod
    def create(
            path_config_supply: Path,
            path_config_demand: Path,
            departure_time_hard_restriction: bool = False,
            action_factor: int = ACTION_FACTOR,
            seed: Union[int, None] = None
    ) -> RobinEnv:
        """
        Create a Robin environment.

        Args:
            path_config_supply (Path): Path to the supply configuration file.
            path_config_demand (Path): Path to the demand configuration file.
            departure_time_hard_restriction (bool): Whether to apply a hard restriction to the departure time.
            seed (int, None): Seed for the random number generator.
            action_factor (int): Factor to multiply the price action.
        
        Returns:
            RobinEnv: Robin environment.
        """
        env = RobinSingleAgentEnv(
            path_config_supply=path_config_supply,
            path_config_demand=path_config_demand,
            departure_time_hard_restriction=departure_time_hard_restriction,
            action_factor=action_factor,
            seed=seed
        )
        env = FlattenObservation(env)
        env = FlattenAction(env)
        return env
