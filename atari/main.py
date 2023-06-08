import math
import random
import numpy as np
import os
import argparse
from collections import deque
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models import *
from memory import *
from atari_wrappers import wrap_deepmind, make_atari

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 1000000
TARGET_UPDATE = 10000
NUM_STEPS = 10000000
M_SIZE = 200000
POLICY_UPDATE = 4
EVALUATE_FREQ = 100000


def optimize_model(train):
    if not train:
        return
    state_batch, action_batch, reward_batch, n_state_batch, done_batch = memory.sample(
        BATCH_SIZE)
    q = policy_net(state_batch).gather(1, action_batch)
    nq = target_net(n_state_batch).max(1)[0].detach()
    expected_state_action_values = (
        nq * GAMMA)*(1.-done_batch[:, 0]) + reward_batch[:, 0]
    loss = F.smooth_l1_loss(q, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def evaluate(step, policy_net, env, n_actions, eps=0.01, num_episode=5):
    best_reward = 0.0
    if not os.path.exists("models"):
        os.makedirs("models")
    env = wrap_deepmind(env)
    agent = Agent(eps, eps, policy_net, EPS_DECAY, n_actions)
    e_rewards = []
    q = deque(maxlen=5)
    for _ in range(num_episode):
        env.reset()
        e_reward = 0
        for _ in range(5):
            n_frame, _, done, _ = env.step(0)
            n_frame = fp(n_frame)
            q.append(n_frame)

        while not done:
            state = torch.cat(list(q))[1:].unsqueeze(0)
            action, eps = agent.select_action(state, train)
            n_frame, reward, done, _ = env.step(action)
            n_frame = fp(n_frame)
            q.append(n_frame)

            e_reward += reward
        e_rewards.append(e_reward)

    f = open(env_name+".csv", 'a')
    avg_reward = float(sum(e_rewards))/float(num_episode)
    std = np.array(e_rewards).std()
    print("The average reward is: %.5f" % (avg_reward,))
    if avg_reward > best_reward:
        torch.save(policy_net.state_dict(), "models/" +
                   env_name+"_"+str(int(avg_reward))+".pth")
        best_reward = avg_reward
    f.write("%f, %f, %d\n" % (avg_reward, std, step))
    f.close()


def fp(n_frame):
    n_frame = torch.from_numpy(n_frame)
    h = n_frame.shape[-2]
    return n_frame.view(1, h, h)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str,
                        default='BreakoutNoFrameskip-v4')
    parser.add_argument("--is_dueling", type=bool, default=True)
    parser.add_argument("--is_double", type=bool, default=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    torch.manual_seed(2023)
    if "NoFrameskip-v4" in args.env_name:
        env_name = args.env_name.replace("NoFrameskip-v4", "")
    env_raw = make_atari(args.env_name)
    env = wrap_deepmind(env_raw)

    c, h, w = fp(env.reset()).shape
    n_actions = env.action_space.n

    if args.is_dueling:
        policy_net = dueling_DQN(h, w, n_actions).to(device)
        target_net = dueling_DQN(h, w, n_actions).to(device)
    else:
        policy_net = DQN(h, w, n_actions).to(device)
        target_net = DQN(h, w, n_actions).to(device)
    policy_net.apply(policy_net.init_weights)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)

    memory = ReplayMemory(M_SIZE, [5, h, w], n_actions)
    agent = Agent(EPS_START, EPS_END, policy_net, EPS_DECAY, n_actions)

    q = deque(maxlen=5)
    rq = deque(maxlen=2)
    done = True
    eps = 0
    episode_len = 0

    for step in range(NUM_STEPS):
        if done:
            env.reset()
            sum_reward = 0
            episode_len = 0
            img, _, _, _ = env.step(1)
            for i in range(5):
                n_frame, reward, _, _ = env.step(0)
                n_frame = fp(n_frame)
                q.append(n_frame)
                rq.append(reward)

        train = len(memory) > 500
        state = torch.cat(list(q))[1:].unsqueeze(0)
        action, eps = agent.select_action(state, train)
        n_frame, reward, done, info = env.step(action)
        n_frame = fp(n_frame)

        q.append(n_frame)
        rq.append(reward)
        memory.push(torch.cat(list(q)).unsqueeze(0), action, rq[0], done)
        episode_len += 1

        if step % POLICY_UPDATE == 0:
            optimize_model(train)

        if step % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if step % EVALUATE_FREQ == 0:
            evaluate(step, policy_net, env_raw,
                     n_actions, eps=0.1, num_episode=20)
