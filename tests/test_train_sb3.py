import numpy as np

from src.robin.rl.entities import RobinEnvFactory

from stable_baselines3 import SAC, TD3
from stable_baselines3.common.env_util import make_vec_env, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import VecNormalize


if __name__ == '__main__':
    # Train then model
    num_cpu = 16
    log_name = 'delete'
    log_dir = f'models/{log_name}'
    vec_env = make_vec_env(
        lambda: RobinEnvFactory.create(path_config_demand='configs/rl/demand_data.yml', path_config_supply='configs/rl/supply_data.yml'),
        n_envs=num_cpu,
        vec_env_cls=SubprocVecEnv
    )
    vec_env = VecNormalize(vec_env, clip_obs=10.0, clip_reward=10.0)
    action_noise = NormalActionNoise(mean=np.zeros(10), sigma=0.1 * np.ones(10))
    model = TD3('MlpPolicy', vec_env, learning_starts=10_000, learning_rate=1e-4, action_noise=action_noise, train_freq=1, gradient_steps=-1, tensorboard_log='models', buffer_size=500_000, verbose=0)
    model.learn(total_timesteps=500_000, tb_log_name=log_name, progress_bar=True)
    model.save(f'{log_dir}_1/{log_name}')
    vec_env.save(f'{log_dir}_1/vec_normalize.pkl')

    # Evaluate the model
    del vec_env, model
    vec_env = make_vec_env(
        lambda: RobinEnvFactory.create(path_config_demand='configs/rl/demand_data.yml', path_config_supply='configs/rl/supply_data.yml'),
        n_envs=num_cpu,
        vec_env_cls=SubprocVecEnv
    )
    vec_env = VecNormalize.load(f'{log_dir}_1/vec_normalize.pkl', vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    model = TD3.load(f'{log_dir}_1/{log_name}')
    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=100)
    print(f'Mean reward: {mean_reward} +/- {std_reward}')
