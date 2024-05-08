from src.robin.rl.entities import FlattenAction, FlattenObservation, RobinEnv

from stable_baselines3 import SAC, DDPG, TD3
from stable_baselines3.common.env_util import make_vec_env, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize


def make_env():
    env = RobinEnv(path_config_demand='configs/rl/demand_data.yml', path_config_supply='configs/rl/supply_data.yml')
    env = FlattenObservation(env)
    env = FlattenAction(env)
    return env


if __name__ == '__main__':
    # Train then model
    num_cpu = 16
    log_name = 'SAC_60_100_BUSINESS_10_NORM_BOTH'
    vec_env = make_vec_env(make_env, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)
    vec_env = VecNormalize(vec_env, clip_obs=10.0, clip_reward=10.0)
    model = SAC('MlpPolicy', vec_env, train_freq=1, gradient_steps=-1, tensorboard_log='models', buffer_size=500_000, verbose=0)
    model.learn(total_timesteps=500_000, tb_log_name=log_name, progress_bar=True)
    model.save(f'models/{log_name}')
    vec_env.save('models/vec_normalize.pkl')

    # Evaluate the model
    del vec_env, model
    vec_env = make_vec_env(make_env, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)
    vec_env = VecNormalize.load('models/vec_normalize.pkl', vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    model = SAC.load(f'models/{log_name}')
    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=100)
    print(f'Mean reward: {mean_reward} +/- {std_reward}')
