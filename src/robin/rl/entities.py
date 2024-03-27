import datetime
import gymnasium as gym

from src.robin.kernel.entities import Kernel
from .constants import *

from copy import copy
from functools import lru_cache
from pathlib import Path
from pettingzoo import ParallelEnv
from pettingzoo.test import parallel_api_test
from typing import Union


class RobinEnv(ParallelEnv):
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
        n_services = len(services)
        print(services)
        print(services_mask)
        # we should check also the capacity?
        # For simplicity, a service only has one price, so just one origin and destination and a seat
        # action masks for each service that is not available (i.e. has no capacity or it has already departed or too soon to put on sale)
        observation_space = gym.spaces.Dict({
            'prices': gym.spaces.Box(low=LOW_PRICE, high=HIGH_PRICE, shape=(n_services,), dtype=float),
            'tickets_sold': gym.spaces.Box(low=LOW_CAPACITY, high=HIGH_CAPACITY, shape=(n_services,), dtype=int),
            'capacities': gym.spaces.Box(low=LOW_CAPACITY, high=HIGH_CAPACITY, shape=(n_services,), dtype=int),
        })
        return observation_space

    @lru_cache(maxsize=None)
    def action_space(self, agent):
        services = self.kernel.supply.filter_services_by_tsp(agent)
        action_space = gym.spaces.Box(
            low=LOW_ACTION, high=HIGH_ACTION, shape=(len(services),), dtype=float
        )
        return action_space


if __name__ == '__main__':
    env = RobinEnv(path_config_demand='configs/rl/demand_data.yml', path_config_supply='configs/rl/supply_data.yml', seed=0)
    env.observation_space(env.possible_agents[0])
    #env.reset()
    #env.render()
    #parallel_api_test(env, num_cycles=1_000)
