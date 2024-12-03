import argparse
import os
import torch

from robin.rl.entities import RobinEnvFactory, StatsSubprocVectorEnv, VectorEnvNormReward

from tianshou.data import Collector
from tianshou.exploration import GaussianNoise
from tianshou.policy import TD3Policy
from tianshou.trainer.utils import test_episode
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic


def get_args() -> argparse.Namespace:
    """
    Get arguments from command line.

    Returns:
        argparse.Namespace: Arguments from command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str)
    parser.add_argument('--path-config-supply', type=str, default='configs/rl/supply_data_connecting.yml')
    parser.add_argument('--path-config-demand', type=str, default='configs/rl/demand_data_connecting.yml')
    parser.add_argument('--algo-name', type=str, default='TD3')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-episodes', type=int, default=10_000)
    parser.add_argument('--buffer-size', type=int, default=1_000_000)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[400, 300])
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--exploration-noise', type=float, default=0.1)
    parser.add_argument('--policy-noise', type=float, default=0.2)
    parser.add_argument('--noise-clip', type=float, default=0.5)
    parser.add_argument('--update-actor-freq', type=int, default=2)
    parser.add_argument('--start-timesteps', type=int, default=5_000)
    parser.add_argument('--epoch', type=int, default=400)
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
    return parser.parse_args()


def test_td3(args: argparse.Namespace = get_args()) -> None:
    """
    Test the TD3 algorithm using the Tianshou library.

    Args:
        args (argparse.Namespace): Arguments from command line.
    """
    # Logging
    log_path = os.path.join(args.input_dir, 'evaluation')
    resume_path = os.path.join(args.input_dir, 'policy.pth')

    # Environment
    torch.manual_seed(args.seed)
    env_fns = [
        lambda: RobinEnvFactory.create(
            path_config_supply=args.path_config_supply,
            path_config_demand=args.path_config_demand,
            multi_agent=False,
            discrete_action_space=False,
            seed=args.seed + i * 100_000
        ) for i in range(args.training_num)
    ]
    env = StatsSubprocVectorEnv(env_fns=env_fns, log_dir=log_path)
    env = VectorEnvNormReward(env)
    env.seed([args.seed + i * 100_000 for i in range(args.training_num)])

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
    policy.load_state_dict(torch.load(resume_path, map_location=args.device))
    print('Loaded agent from: ', resume_path)

    # Collector
    collector = Collector(policy, env)
    collector.reset()

    # Test episodes
    test_episode(policy, collector, test_fn=None, epoch=None, n_episode=args.n_episodes)


if __name__ == '__main__':
    test_td3()
