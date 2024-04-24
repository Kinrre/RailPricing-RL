from src.robin.rl.entities import FlattenAction, FlattenObservation, RobinEnv

env = RobinEnv(path_config_demand='configs/rl/demand_data.yml', path_config_supply='configs/rl/supply_data.yml', seed=0)
env = FlattenObservation(env)
env = FlattenAction(env)
print(env.observation_space)
print(env.action_space)

observation, info = env.reset(seed=0)
episodic_reward = 0
action = env.action_space.sample()

for _ in range(10_000):
    observation, reward, terminated, truncated, info = env.step(action)
    episodic_reward += reward

    if terminated or truncated:
        observation, info = env.reset(seed=0)
        print(f'Episodic reward: {episodic_reward}')
        episodic_reward = 0
env.close()
