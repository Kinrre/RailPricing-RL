from src.robin.kernel.entities import Kernel
from .constants import *

from gymnasium import ActionWrapper, Env
from gymnasium import spaces
from gymnasium.spaces.utils import flatten_space, flatten, unflatten
from gymnasium.wrappers import FlattenObservation


from copy import copy
from functools import lru_cache
from pathlib import Path
from pettingzoo import ParallelEnv
from pettingzoo.test import parallel_api_test
from typing import Union


class FlattenAction(ActionWrapper):
    """Action wrapper that flattens the action space."""
    def __init__(self, env):
        super().__init__(env)
        self.action_space = flatten_space(self.env.action_space)
        
    def action(self, action):
        return flatten(self.env.action_space, action)

    def reverse_action(self, action):
        return unflatten(self.env.action_space, action)


class RobinEnv(Env):

    def __init__(self, path_config_supply: Path, path_config_demand: Path, seed: Union[int, None] = None) -> None:
        self.path_config_supply = path_config_supply
        self.path_config_demand = path_config_demand
        self.kernel = Kernel(self.path_config_supply, self.path_config_demand, seed)

    def _get_obs(self):
        # next step, get the obs from the supply
        pass

    def _get_info(self) -> dict:
        return {}

    def step(self, action: List[Dict]):
        # simulate a day
        # calculate the reward
        pass

    def reset(self, seed: Union[int, None] = None) -> None:
        super().reset(seed=seed)
        self.kernel = Kernel(self.path_config_supply, self.path_config_demand, seed)

    # cache
    @property
    def observation_space(self) -> spaces.Space:
        observation_space = spaces.Tuple([
            spaces.Dict({
                # service already departed?
                # date time details?
                'line': spaces.Discrete(self.n_lines),
                'corridor': spaces.Discrete(self.n_corridors),
                'time_slot': spaces.Discrete(self.n_time_slots),
                'rolling_stock': spaces.Discrete(self.n_rolling_stocks),
                # capacity of the rolling stock?
                'prices': spaces.Tuple([
                    spaces.Dict({
                        'origin': spaces.Discrete(self.n_stops),
                        'destination': spaces.Discrete(self.n_stops),
                        'seats': spaces.Tuple([
                            spaces.Dict({
                                'seat_type': spaces.Discrete(self.n_seats),
                                'price': spaces.Box(low=0, high=np.inf, shape=())
                            }) for _ in range(self.n_seats)
                        ])
                    }) for _ in range(self.n_stops - 1) # there aren't prices for the same origin and destination
                ]),
                'tickets_sold': spaces.Tuple([
                    spaces.Dict({
                        'origin': spaces.Discrete(self.n_stops),
                        'destination': spaces.Discrete(self.n_stops),
                        'count': spaces.Discrete(self.max_capacity) # split into seat types?
                    }) for _ in range(self.n_stops - 1) # there aren't prices for the same origin and destination
                ]),
            }) for _ in range(self.n_services)
        ])
        return observation_space

    @property
    def action_space(self) -> spaces.Space:
        action_space = spaces.Tuple([
            spaces.Dict({
                'origin': spaces.Discrete(self.n_stops),
                'destination': spaces.Discrete(self.n_stops),
                'seats': spaces.Tuple([
                    spaces.Dict({
                        'seat_type': spaces.Discrete(self.n_seats),
                        'price': spaces.Box(low=0, high=1, shape=()) # normalization of price modifications
                    }) for _ in range(self.n_seats)
                ])
            }) for _ in range(self.n_services)
        ])
        return action_space

    @property
    def n_services(self):
        """Number of services."""
        return len(self.kernel.supply.services)
    
    @property
    def n_lines(self):
        """Number of lines."""
        return len(self.kernel.supply.lines)
    
    @property
    def n_corridors(self):
        """Number of corridors."""
        return len(self.kernel.supply.corridors)
    
    @property
    def n_time_slots(self):
        """Number of time slots."""
        return len(self.kernel.supply.time_slots)
    
    @property
    def n_rolling_stocks(self):
        """Number of rolling stocks."""
        return len(self.kernel.supply.rolling_stocks)

    @property
    def n_seats(self):
        """Number of seat types."""
        return len(self.kernel.supply.seats)
    
    @property
    def n_stops(self):
        """Maximum number of stops in a line."""
        line = max(self.kernel.supply.lines, key=lambda x: len(x.stations))
        return len(line.stations)

    @property
    def max_capacity(self):
        """Maximum capacity of a rolling stock."""
        return max(self.kernel.supply.rolling_stocks, key=lambda x: x.total_capacity).total_capacity


class RobinMarlEnv(ParallelEnv):
    metadata = {
        'name': 'RobinEnv_v0',
    }

    def __init__(self, path_config_supply: Path, path_config_demand: Path, seed: Union[int, None] = None) -> None:
        self.kernel = Kernel(path_config_supply, path_config_demand, seed)
        self.possible_agents = [tsp.id for tsp in self.kernel.supply.tsps]

    def _get_obs(self):
        observations = {}
        for agent in self.agents:
            services = self.kernel.supply.filter_services_by_tsp(agent)
            # For simplicity, only first seat is considered
            seat = self.kernel.supply.seats[0]
            prices = [list(service.prices.values())[0][seat] for service in services]
            tickets_sold = [list(service.tickets_sold_seats.values())[0] for service in services]
            capacities = [service.rolling_stock.total_capacity for service in services]
            observations[agent] = {
                'prices': prices,
                'tickets_sold': tickets_sold,
                'capacities': capacities,
            }
        return observations

    def _get_truncations(self):
        return {agent: self._elapsed_steps >= self._max_episode_steps for agent in self.agents}

    def _get_infos(self):
        return {agent: {} for agent in self.agents}

    def step(self, actions):
        observations = self._get_obs()
        rewards = {agent: 0 for agent in self.agents}
        terminations = self._get_truncations()
        truncations = self._get_truncations()
        infos = self._get_infos()
        return observations, rewards, terminations, truncations, infos

    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        observations = self._get_obs()
        infos = self._get_infos()
        return observations, infos

    def render(self):
        observations = self._get_obs()
        print(observations)

    @lru_cache(maxsize=None)
    def observation_space(self, agent):
        services = self.kernel.supply.filter_services_by_tsp(tsp_id=agent)
        active_services = self.kernel.supply.filter_active_services_by_tsp(tsp_id=agent, date=self.kernel.simulation_day)
        services_mask = np.isin(services, active_services).astype(int)
        # we should check also the capacity?
        # For simplicity, a service only has one price, so just one origin and destination and a seat
        # action masks for each service that is not available (i.e. has no capacity or it has already departed or too soon to put on sale)
        n_services = len(services)
        n_stations = len(self.kernel.supply.stations)
        n_lines = len(self.kernel.supply.lines)
        n_corridors = len(self.kernel.supply.corridors)
        observation_space = spaces.Tuple([
            spaces.Dict({
                'service': spaces.Discrete(n_services),
                'line': spaces.Discrete(n_lines),
                'corridor': spaces.Discrete(n_corridors),

                'origin': spaces.Discrete(n_stations),
                'destination': spaces.Discrete(n_stations),


                'prices': spaces.Box(low=LOW_PRICE, high=HIGH_PRICE, shape=(n_services,), dtype=float),
                'tickets_sold': spaces.Box(low=LOW_TICKETS_SOLD, high=HIGH_TICKETS_SOLD, shape=(n_services,), dtype=int),
                'capacities': spaces.Box(low=LOW_CAPACITY, high=HIGH_CAPACITY, shape=(n_services,), dtype=int),
                'action_mask': spaces.Box(low=LOW_ACTION_MASK, high=HIGH_ACTION_MASK, shape=(n_services,), dtype=int),
            }) for _ in range(n_services)
        ])
        sample = observation_space.sample()
        return observation_space

    @lru_cache(maxsize=None)
    def action_space(self, agent):
        services = self.kernel.supply.filter_services_by_tsp(agent)
        action_space = spaces.Box(
            low=LOW_ACTION, high=HIGH_ACTION, shape=(len(services),), dtype=float
        )
        return action_space


if __name__ == '__main__':
    env = RobinEnv(path_config_demand='configs/demand_data.yml', path_config_supply='configs/supply_data.yml', seed=0)
    env = FlattenObservation(env)
    env = FlattenAction(env)
    print(env.observation_space.sample())
    print(env.observation_space)
    print(env.action_space.sample())
    print(env.action_space)
