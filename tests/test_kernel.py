import os

from robin.kernel.entities import Kernel

os.makedirs('data/kernel_output', exist_ok=True)

path_config_supply = 'configs/rl/supply_data_connecting.yml'
path_config_demand = 'configs/rl/demand_data_connecting.yml'
seed = 0

kernel = Kernel(path_config_supply, path_config_demand, seed)
services = kernel.simulate('data/kernel_output/output.csv', departure_time_hard_restriction=True)

for service in services:
    print(service)
