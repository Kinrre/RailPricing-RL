"""Utils module for RL algorithms."""

from robin.rl.entities import RobinEnvFactory, StatsSubprocVectorEnv
from robin.rl.algorithms.constants import IS_COOPERATIVE

def create_env(
        supply_config: str,
        demand_config: str,
        algorithm: str,
        seed: int,
        seed_rank_multiplier: int,
        n_workers: int,
        run_name: str
    ) -> StatsSubprocVectorEnv:
    """
    Create the ROBIN environment.

    Args:
        supply_config (str): Supply configuration file.
        demand_config (str): Demand configuration file.
        algorithm (str): Algorithm to use.
        seed (int): Seed of the experiment.
        seed_rank_multiplier (int): Seed rank multiplier.
        n_workers (int): Number of workers.
        run_name (str): Name of the run.
        
    Returns:
        StatsSubprocVectorEnv: The ROBIN environment.
    """
    env_fns = [
        lambda: RobinEnvFactory.create(
            path_config_supply=supply_config,
            path_config_demand=demand_config,
            multi_agent=True,
            cooperative=IS_COOPERATIVE[algorithm],
            discrete_action_space=False,
            seed=seed + i * seed_rank_multiplier
        ) for i in range(n_workers)
    ]
    env = StatsSubprocVectorEnv(env_fns=env_fns, log_dir=run_name)
    env.seed([seed + i * seed_rank_multiplier for i in range(n_workers)])
    return env
