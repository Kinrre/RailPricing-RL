import numpy as np

from gymnasium import Env
from robin.rl.entities import RobinEnvFactory

DEFAULT_CONFIG_DEMAND = 'configs/rl/demand_data_connecting.yml'
DEFAULT_CONFIG_SUPPLY = 'configs/rl/supply_data_connecting.yml'
MULTI_AGENT = True
COOPERATIVE = False
DEFAULT_NUM_STEPS = 1_000
SEED = 0


def test_env(env: Env, num_steps: int = DEFAULT_NUM_STEPS) -> None:
    """
    Test the environment with a random agent.

    Args:
        env (Env): The environment to test.
        num_steps (int): The number of steps to run the environment.
    """
    observation, info = env.reset()
    rewards = []
    episodic_reward = 0

    for i in range(num_steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        episodic_reward += reward

        if terminated or truncated:
            observation, info = env.reset()
            rewards.append(episodic_reward)
            episodic_reward = 0

    env.close()
    print(f'Mean episodic reward (random agent): {np.mean(rewards)} +/- {np.std(rewards)}')


def test_multi_agent_env(env: Env, num_steps: int = DEFAULT_NUM_STEPS) -> None:
    """
    Test the multi-agent environment with random agents.

    Args:
        env (Env): The environment to test.
        num_steps (int): The number of steps to run the environment.
    """
    num_agents = env.unwrapped.num_agents
    observation, info = env.reset()
    rewards = [[] for _ in range(num_agents)]
    episodic_reward = np.zeros((num_agents), dtype=np.float32)

    for i in range(num_steps):
        actions = [action_space.sample() for action_space in env.action_space]
        observation, reward, terminated, truncated, info = env.step(actions)
        episodic_reward += reward

        if terminated.all() or truncated.all():
            observation, info = env.reset()
            for i in range(num_agents):
                rewards[i].append(episodic_reward[i])
            episodic_reward = np.zeros((num_agents), dtype=np.float32)

    env.close()
    for i, agent in enumerate(env.unwrapped.agents):
        print(f'{agent} - Mean episodic reward (random agent): {np.mean(rewards[i])} +/- {np.std(rewards[i])}')


if __name__ == '__main__':
    env = RobinEnvFactory.create(
        path_config_supply=DEFAULT_CONFIG_SUPPLY,
        path_config_demand=DEFAULT_CONFIG_DEMAND,
        multi_agent=MULTI_AGENT,
        cooperative=COOPERATIVE,
        seed=SEED
    )
    print(f'Number of services: {len(env.unwrapped.kernel.supply.services)}')
    print(env.observation_space)
    print(env.action_space)
    
    if MULTI_AGENT:
        test_multi_agent_env(env)
    else:
        test_env(env)
