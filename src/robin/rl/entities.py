from src.robin.kernel.entities import Kernel
from .constants import LOW_PRICE, HIGH_PRICE, LOW_ACTION, HIGH_ACTION

from functools import cached_property, lru_cache
from gymnasium import ActionWrapper, Env
from gymnasium import spaces
from gymnasium.spaces.utils import flatten_space, flatten, unflatten
from gymnasium.wrappers import FlattenObservation
from pathlib import Path
from typing import Tuple, Union
from pprint import pprint


class FlattenAction(ActionWrapper):
    """Action wrapper that flattens the action space."""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = flatten_space(self.env.action_space)
        
    def action(self, action):
        return unflatten(self.env.action_space, action)

    def reverse_action(self, action):
        return flatten(self.env.action_space, action)


class FlattenObservation(FlattenObservation):
    pass


class RobinEnv(Env):
    """
    Reinforcement learning environment for the Robin simulator.

    Attributes:
        path_config_supply (Path): Path to the supply configuration file.
        path_config_demand (Path): Path to the demand configuration file.
        kernel (Kernel): Kernel of the simulator.
    """

    def __init__(self, path_config_supply: Path, path_config_demand: Path, seed: Union[int, None] = None) -> None:
        """
        Initialize the environment.

        Args:
            path_config_supply (Path): Path to the supply configuration file.
            path_config_demand (Path): Path to the demand configuration file.
            seed (int, None): Seed for the random number generator.
        """
        self.path_config_supply = path_config_supply
        self.path_config_demand = path_config_demand
        self.kernel = Kernel(self.path_config_supply, self.path_config_demand, seed)

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

    def _get_obs(self) -> list:
        """
        Get the observation of the environment.

        Returns:
            list: Observation of the environment.
        """
        obs = []
        for service in self.kernel.supply.services:
            obs.append({
                'line': self._get_element_idx_from_id(self.kernel.supply.lines, service.line.id),
                'corridor': self._get_element_idx_from_id(self.kernel.supply.corridors, service.line.corridor.id),
                'time_slot': self._get_element_idx_from_id(self.kernel.supply.time_slots, service.time_slot.id),
                'rolling_stock': self._get_element_idx_from_id(self.kernel.supply.rolling_stocks, service.rolling_stock.id),
                'prices': [{
                    'origin': self._get_element_idx_from_id(self.kernel.supply.stations, origin),
                    'destination': self._get_element_idx_from_id(self.kernel.supply.stations, destination),
                    'seats': [{
                        'seat_type': self._get_element_idx_from_id(self.kernel.supply.seats, seat.id),
                        'price': price
                    } for seat, price in seats.items()]
                } for (origin, destination), seats in service.prices.items()],
                'tickets_sold': [{
                    'origin': self._get_element_idx_from_id(self.kernel.supply.stations, origin),
                    'destination': self._get_element_idx_from_id(self.kernel.supply.stations, destination),
                    'seats': [{
                        'seat_type': self._get_element_idx_from_id(self.kernel.supply.seats, seat.id),
                        'count': count
                    } for seat, count in seats.items()]
                } for (origin, destination), seats in service.tickets_sold_pair_seats.items()]
            })
        return obs

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

        The total profit of the services.

        Returns:
            float: Reward of the environment.
        """
        # think about how we can feed the reward in different components to the agent
        # as we have multiple services, markets and seats, this information can be useful
        reward = sum(service.total_profit for service in self.kernel.supply.services)
        return reward

    def step(self, action: list) -> Tuple[list, float, bool, bool, dict]:
        # simulate a day
        pprint(action, sort_dicts=False)
        print(self._get_reward())
        obs = self._get_obs()
        reward = self._get_reward()
        terminated = False
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
        self.kernel = Kernel(self.path_config_supply, self.path_config_demand, seed)
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    @cached_property
    def observation_space(self) -> spaces.Space:
        """
        Observation space of the environment.

        Returns:
            spaces.Space: Observation space of the environment.
        """
        observation_space = spaces.Tuple([
            spaces.Dict({
                # service already departed for an action mask?
                # date time details? day of the week?
                # capacity of the rolling stock?
                'line':spaces.Discrete(1, start=self._get_element_idx_from_id(self.kernel.supply.lines, service.line.id)),
                'corridor': spaces.Discrete(1, start=self._get_element_idx_from_id(self.kernel.supply.corridors, service.line.corridor.id)),
                'time_slot': spaces.Discrete(1, start=self._get_element_idx_from_id(self.kernel.supply.time_slots, service.time_slot.id)),
                'rolling_stock': spaces.Discrete(1, start=self._get_element_idx_from_id(self.kernel.supply.rolling_stocks, service.rolling_stock.id)),
                'prices': spaces.Tuple([
                    spaces.Dict({
                        'origin': spaces.Discrete(1, start=self._get_element_idx_from_id(self.kernel.supply.stations, origin)),
                        'destination': spaces.Discrete(1, start=self._get_element_idx_from_id(self.kernel.supply.stations, destination)),
                        'seats': spaces.Tuple([
                            spaces.Dict({
                                'seat_type': spaces.Discrete(1, start=self._get_element_idx_from_id(self.kernel.supply.seats, seat.id)),
                                'price': spaces.Box(low=LOW_PRICE, high=HIGH_PRICE, shape=())
                            }) for seat, _ in seats.items()
                        ])
                    }) for (origin, destination), seats in service.prices.items()
                ]),
                'tickets_sold': spaces.Tuple([
                    spaces.Dict({
                        'origin': spaces.Discrete(1, start=self._get_element_idx_from_id(self.kernel.supply.stations, origin)),
                        'destination': spaces.Discrete(1, start=self._get_element_idx_from_id(self.kernel.supply.stations, destination)),
                        'seats': spaces.Tuple([
                            spaces.Dict({
                                'seat_type': spaces.Discrete(1, start=self._get_element_idx_from_id(self.kernel.supply.seats, seat.id)),
                                'count': spaces.Discrete(service.rolling_stock.total_capacity)
                            }) for seat, _ in seats.items()
                        ])
                    }) for (origin, destination), seats in service.prices.items()
                ]),
            }) for service in self.kernel.supply.services
        ])
        return observation_space

    @cached_property
    def action_space(self) -> spaces.Space:
        """
        Action space of the environment.

        Returns:
            spaces.Space: Action space of the environment.
        """
        action_space = spaces.Tuple([
            spaces.Dict({
                'prices': spaces.Tuple([
                    spaces.Dict({
                        'origin': spaces.Discrete(1, start=self._get_element_idx_from_id(self.kernel.supply.stations, origin)),
                        'destination': spaces.Discrete(1, start=self._get_element_idx_from_id(self.kernel.supply.stations, destination)),
                        'seats': spaces.Tuple([
                            spaces.Dict({
                                'seat_type': spaces.Discrete(1, start=self._get_element_idx_from_id(self.kernel.supply.seats, seat.id)),
                                'price': spaces.Box(low=LOW_ACTION, high=HIGH_ACTION, shape=()) # normalization of price modifications
                            }) for seat, _ in seats.items()
                        ])
                    }) for (origin, destination), seats in service.prices.items()
                ])
            }) for service in self.kernel.supply.services
        ])
        return action_space
