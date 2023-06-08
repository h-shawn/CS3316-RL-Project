import os
import argparse
from datetime import datetime
import gym
from agent import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_STEPS = 1000000


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default='Hopper-v2')
    parser.add_argument('--method', choices=['PPO', 'SAC'], default='PPO')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    env = gym.make(args.env_name)
    local_dir = os.path.join(args.env_name, f'{args.method}')
    if args.method == 'SAC':
        agent = SAC(env, True, True, NUM_STEPS, local_dir)
    else:
        agent = PPO(env, NUM_STEPS, local_dir)
    agent.run()
