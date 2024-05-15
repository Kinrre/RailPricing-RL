import numpy as np

from src.robin.rl.entities import FlattenAction, FlattenObservation, RobinEnv


env = RobinEnv(path_config_demand='configs/rl/demand_data.yml', path_config_supply='configs/rl/supply_data.yml')
print(f'Number of services: {len(env.kernel.supply.services)}')
env = FlattenObservation(env)
env = FlattenAction(env)
print(env.observation_space)
print(env.action_space)

observation, info = env.reset()
rewards = []
episodic_reward = 0

for _ in range(1_000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    episodic_reward += reward

    if terminated or truncated:
        observation, info = env.reset()
        rewards.append(episodic_reward)
        episodic_reward = 0

env.close()
print(f'Mean episodic reward (random agent): {np.mean(rewards)} +/- {np.std(rewards)}')
