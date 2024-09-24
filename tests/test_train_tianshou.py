import argparse
import datetime
import os
import pprint
import torch
import yaml

from src.robin.rl.entities import RobinEnvFactory, StatsSubprocVectorEnv, VectorEnvNormObsReward

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv, VectorEnvNormObs
from tianshou.exploration import GaussianNoise
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.policy import TD3Policy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic


def get_args() -> argparse.Namespace:
    """
    Get arguments from command line.

    Returns:
        argparse.Namespace: Arguments from command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Tianshou_Business')
    parser.add_argument('--path-config-supply', type=str, default='configs/rl/supply_data_connecting.yml')
    parser.add_argument('--path-config-demand', type=str, default='configs/rl/demand_data_connecting.yml')
    parser.add_argument('--algo-name', type=str, default='TD3')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=500_000)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[400, 300])
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=1e-3)
    parser.add_argument('--exploration-noise', type=float, default=0.1)
    parser.add_argument('--policy-noise', type=float, default=0.2)
    parser.add_argument('--noise-clip', type=float, default=0.5)
    parser.add_argument('--update-actor-freq', type=int, default=2)
    parser.add_argument('--start-timesteps', type=int, default=10_000)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--step-per-epoch', type=int, default=2_500)
    parser.add_argument('--step-per-collect', type=int, default=1)
    parser.add_argument('--update-per-step', type=int, default=1)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--training-num', type=int, default=16)
    parser.add_argument('--test-num', type=int, default=10*16)
    parser.add_argument('--logdir', type=str, default='models')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
    )
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--resume-id', type=str, default=None)
    return parser.parse_args()


def log_args_and_git_commit(log_path: str, args: argparse.Namespace) -> None:
    """
    Log arguments and git commit to the log directory.

    Args:
        log_path (str): Path to the log directory.
        args (argparse.Namespace): Arguments from command line.
    """
    with open(os.path.join(log_path, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f)
    os.system(f'git -C {os.path.dirname(os.path.abspath(__file__))} log -1 > {log_path}/commit.txt')
    os.system(f'git -C {os.path.dirname(os.path.abspath(__file__))} diff > {log_path}/diff.patch')


def test_td3(args: argparse.Namespace = get_args()) -> None:
    """
    Test the TD3 algorithm using the Tianshou library.

    Args:
        args (argparse.Namespace): Arguments from command line.
    """
    # Logging
    now = datetime.datetime.now().strftime('%d%m%y-%H%M%S')
    log_name = os.path.join(args.task, args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)

    logger_factory = LoggerFactoryDefault()
    logger_factory.logger_type = 'tensorboard'
    logger = logger_factory.create_logger(
        log_dir=log_path,
        experiment_name=log_name,
        run_id=args.resume_id,
        config_dict=vars(args),
    )
    log_args_and_git_commit(log_path, args)

    def save_best_fn(policy: BasePolicy) -> None:
        """
        Save the best policy to the log directory and the environment statistics.

        Args:
            policy (BasePolicy): Policy to save.
        """
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))
        torch.save(env.obs_rms, os.path.join(log_path, 'obs_rms.pth'))
        torch.save(env.reward_rms, os.path.join(log_path, 'reward_rms.pth'))

    # Environment
    torch.manual_seed(args.seed)
    env_fns = [
        lambda: RobinEnvFactory.create(
            path_config_supply=args.path_config_supply,
            path_config_demand=args.path_config_demand,
            multi_agent=False,
            discrete_action_space=False,
            seed=args.seed + i * 1000
        ) for i in range(args.training_num)
    ]
    env = SubprocVectorEnv(env_fns=env_fns)
    env = VectorEnvNormObsReward(env)
    test_env = StatsSubprocVectorEnv(env_fns=env_fns, log_dir=log_path)
    test_env = VectorEnvNormObs(test_env, update_obs_rms=False)
    test_env.set_obs_rms(env.obs_rms)

    # Print environment information
    obs_space = env.observation_space[0]
    obs_shape = env.observation_space[0].shape
    action_space = env.action_space[0]
    action_shape = env.action_space[0].shape
    min_action = env.action_space[0].low[0]
    max_action = env.action_space[0].high[0]
    print(f'Observation space: {obs_space}')
    print(f'Observation shape: {obs_shape}')
    print(f'Action space: {action_space}')
    print(f'Action shape: {action_shape}')
    print(f'Action low: {min_action}')
    print(f'Action high: {max_action}')

    # Actor
    net_actor = Net(
        state_shape=obs_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device
    )
    actor = Actor(
        preprocess_net=net_actor,
        action_shape=action_shape,
        max_action=max_action,
        device=args.device
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    # Critic
    net_critic1 = Net(
        state_shape=obs_shape[0] + action_shape[0],
        #action_shape=obs_shape,
        hidden_sizes=args.hidden_sizes,
        concat=False,
        device=args.device,
    )
    net_critic2 = Net(
        state_shape=obs_shape[0] + action_shape[0],
        #action_shape=obs_shape,
        hidden_sizes=args.hidden_sizes,
        concat=False,
        device=args.device,
    )
    critic1 = Critic(net_critic1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_critic2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    # Policy
    policy: TD3Policy = TD3Policy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic1,
        critic_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        policy_noise=args.policy_noise,
        update_actor_freq=args.update_actor_freq,
        noise_clip=args.noise_clip,
        estimation_step=args.n_step,
        action_space=action_space,
    )
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print('Loaded agent from: ', args.resume_path)
    print(policy)

    # Collector
    buffer: VectorReplayBuffer | ReplayBuffer
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(env))
    else:
        buffer = ReplayBuffer(args.buffer_size)
    train_collector = Collector(policy, env, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_env)
    train_collector.reset()
    train_collector.collect(n_step=args.start_timesteps, random=True)

    # Trainer
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        save_best_fn=save_best_fn,
        logger=logger,
        update_per_step=args.update_per_step,
        test_in_train=False
    ).run()
    pprint.pprint(result)


if __name__ == '__main__':
    test_td3()
