from src.robin.kernel.entities import Kernel
from .constants import *

from gymnasium import ActionWrapper, Env
from gymnasium import spaces
from gymnasium.spaces.utils import flatten_space, flatten, unflatten
from gymnasium.wrappers import FlattenObservation

from pathlib import Path
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

    def _get_obs(self) -> list:
        # next step, get the obs from the supply
        obs = []
        for service in self.kernel.supply.services:
            obs.append({
                'line': service.line.id,
                'corridor': service.line.id,
                'time_slot': service.time_slot.id,
                'rolling_stock': service.rolling_stock.id,
                'prices': [{
                    'origin': price.origin.id,
                    'destination': price.destination.id,
                    'seats': [{
                        'seat_type': seat.id,
                        'price': price.get_price(seat)
                    } for seat in self.kernel.supply.seats]
                } for price in service.prices],
                'tickets_sold': [{
                    'origin': ticket.origin.id,
                    'destination': ticket.destination.id,
                    'count': ticket.count
                } for ticket in service.tickets_sold]
            })

    def _get_info(self) -> dict:
        return {}

    def step(self, action: list):
        # simulate a day
        # calculate the reward
        pass

    def reset(self, seed: Union[int, None] = None) -> None:
        super().reset(seed=seed)
        self.kernel = Kernel(self.path_config_supply, self.path_config_demand, seed)
        self._get_obs()

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


if __name__ == '__main__':
    env = RobinEnv(path_config_demand='configs/demand_data.yml', path_config_supply='configs/supply_data.yml', seed=0)
    env = FlattenObservation(env)
    env = FlattenAction(env)
    print(env.observation_space.sample())
    print(env.observation_space)
    print(env.action_space.sample())
    print(env.action_space)
