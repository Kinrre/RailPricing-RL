from src.robin.rl.entities import FlattenAction, FlattenObservation, RobinEnv

env = RobinEnv(path_config_demand='configs/demand_data.yml', path_config_supply='configs/supply_data.yml', seed=0)
env = FlattenObservation(env)
env = FlattenAction(env)
print(env.observation_space)
print(env.action_space)

observation, info = env.reset()
episodic_reward = 0
for _ in range(10_000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    episodic_reward += reward

    if terminated or truncated:
        observation, info = env.reset()
        print(f'Episodic reward: {episodic_reward}')
env.close()
