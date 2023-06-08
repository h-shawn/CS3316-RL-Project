import os
import random
import numpy as np
import torch
from torch.optim import Adam
from rltorch.memory import MultiStepMemory, PrioritizedMemory

from models import *
from memory import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_batch(state, action, reward, next_state, done, device):
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    action = torch.FloatTensor([action]).view(1, -1).to(device)
    reward = torch.FloatTensor([reward]).unsqueeze(0).to(device)
    next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
    done = torch.FloatTensor([done]).unsqueeze(0).to(device)
    return state, action, reward, next_state, done


def update_params(optimizer, network, loss, grad_clip=None, retain_graph=False):
    optimizer.zero_grad()
    loss.backward(retain_graph=retain_graph)
    if grad_clip is not None:
        for p in network.modules():
            torch.nn.utils.clip_grad_norm_(p.parameters(), grad_clip)
    optimizer.step()


def convert_to_tensor(*value):
    device = value[0]
    return [torch.tensor(x).float().to(device) for x in value[1:]]


def make_transition(state, action, reward, next_state, done, log_prob=None):
    transition = {}
    transition['state'] = state
    transition['action'] = action
    transition['reward'] = reward
    transition['next_state'] = next_state
    transition['log_prob'] = log_prob
    transition['done'] = done
    return transition


def make_mini_batch(*value):
    mini_batch_size = value[0]
    full_batch_size = len(value[1])
    full_indices = np.arange(full_batch_size)
    np.random.shuffle(full_indices)
    for i in range(full_batch_size // mini_batch_size):
        indices = full_indices[mini_batch_size * i: mini_batch_size * (i + 1)]
        yield [x[indices] for x in value[1:]]


class PPO:
    def __init__(self, env, n_steps, local_dir=None):
        self.env = env
        self.num_steps = n_steps
        self.local_dir = local_dir
        self.model_dir = os.path.join(local_dir, "model")
        os.makedirs(self.model_dir, exist_ok=True)

        self.batch_size = 64
        self.traj_len = 2000
        self.train_epoch = 10
        self.ent_coef = 0.01
        self.critic_coef = 0.5
        self.lr = 1e-4
        self.max_grad_norm = 0.5
        self.max_clip = 0.2
        self.reward_scaling = 0.1
        self.log_interval = 10
        self.gamma = 0.99
        self.lam = 0.95

        self.policy = PPOActor(self.env.observation_space.shape[0],
                               self.env.action_space.shape[0],
                               hidden_size=256).to(device)
        self.critic = VCritic(
            self.env.observation_space.shape[0], 256).to(device)

        self.policy_optimizer = Adam(self.policy.parameters(), lr=self.lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr)

        self.memory = ReplayBuffer(
            True, self.traj_len, self.env.observation_space.shape[0], self.env.action_space.shape[0])

        self.train_rewards = RunningMeanStats(self.log_interval)
        self.state_rms = RunningMeanStd(env.observation_space.shape[0])
        self.steps = 0
        self.episodes = 0
        self.best_reward = 0.

    def get_gae(self, states, rewards, next_states, dones):
        values = self.critic(states).detach()
        td_target = rewards + self.gamma * \
            self.critic(next_states) * (1 - dones)
        delta = td_target - values
        delta = delta.detach().cpu().numpy()
        advantage_lst = []
        advantage = 0.0
        for idx in reversed(range(len(delta))):
            if dones[idx] == 1:
                advantage = 0.0
            advantage = self.gamma * self.lam * advantage + delta[idx][0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantages = torch.tensor(
            advantage_lst, dtype=torch.float).to(device)
        return values, advantages

    def optimize_model(self):
        data = self.memory.sample(shuffle=False)
        states, actions, rewards, next_states, dones, old_log_probs = convert_to_tensor(device, data['state'],
                                                                                        data['action'], data['reward'],
                                                                                        data['next_state'],
                                                                                        data['done'], data['log_prob'])
        old_values, advantages = self.get_gae(
            states, rewards, next_states, dones)
        returns = advantages + old_values
        advantages = (advantages - advantages.mean())/(advantages.std()+1e-3)

        for i in range(self.train_epoch):
            for state, action, advantage, return_, old_value, old_log_prob \
                    in make_mini_batch(self.batch_size, states, actions, advantages, returns, old_values, old_log_probs):
                curr_mu, curr_sigma = self.policy(state)
                value = self.critic(state).float()
                curr_dist = torch.distributions.Normal(curr_mu, curr_sigma)
                entropy = curr_dist.entropy() * self.ent_coef
                curr_log_prob = curr_dist.log_prob(action).sum(1, keepdim=True)

                ratio = torch.exp(curr_log_prob - old_log_prob.detach())
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.max_clip,
                                    1+self.max_clip) * advantage
                actor_loss = (-torch.min(surr1, surr2)-entropy).mean()

                old_value_clipped = old_value + \
                    (value - old_value).clamp(-self.max_clip, self.max_clip)
                value_loss = (value - return_.detach().float()).pow(2)
                value_loss_clipped = (
                    old_value_clipped - return_.detach().float()).pow(2)
                critic_loss = 0.5 * self.critic_coef * \
                    torch.max(value_loss, value_loss_clipped).mean()

                self.policy_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

    def run(self):
        episode_reward = 0.
        state_lst = []
        state_ = (self.env.reset())
        state = np.clip((state_ - self.state_rms.mean) /
                        (self.state_rms.var ** 0.5 + 1e-8), -5, 5)
        while True:
            for t in range(self.traj_len):
                state_lst.append(state_)
                mu, sigma = self.policy(torch.from_numpy(
                    state).float().to(device))
                dist = torch.distributions.Normal(mu, sigma[0])
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1, keepdim=True)
                next_state_, reward, done, _ = self.env.step(
                    action.cpu().numpy())
                next_state = np.clip(
                    (next_state_ - self.state_rms.mean) / (self.state_rms.var ** 0.5 + 1e-8), -5, 5)
                self.steps += 1
                episode_reward += reward
                transition = make_transition(state, action.cpu().numpy(), np.array([reward * self.reward_scaling]),
                                             next_state, np.array([done]), log_prob.detach().cpu().numpy())
                self.memory.append(transition)
                state_ = next_state_
                state = next_state

                if done:
                    self.train_rewards.append(episode_reward)
                    self.episodes += 1
                    episode_reward = 0.
                    state_ = (self.env.reset())
                    state = np.clip((state_ - self.state_rms.mean) /
                                    (self.state_rms.var ** 0.5 + 1e-8), -5, 5)

            self.optimize_model()
            self.state_rms.update(np.vstack(state_lst))
            self.evaluate()

            if self.steps > self.num_steps:
                break

    def evaluate(self):
        episodes = 10
        returns = np.zeros((episodes,), dtype=np.float32)

        for i in range(episodes):
            state_ = (self.env.reset())
            state = np.clip((state_ - self.state_rms.mean) /
                            (self.state_rms.var ** 0.5 + 1e-8), -5, 5)
            episode_reward = 0.
            done = False
            while not done:
                mu, sigma = self.policy(torch.FloatTensor(
                    state).unsqueeze(0).to(device))
                dist = torch.distributions.Normal(mu, sigma[0])
                action = dist.sample().cpu().numpy().reshape(-1)
                next_state_, reward, done, _ = self.env.step(action)
                next_state = np.clip(
                    (next_state_ - self.state_rms.mean) / (self.state_rms.var ** 0.5 + 1e-8), -5, 5)
                episode_reward += reward
                state = next_state
            returns[i] = episode_reward

        mean_return = float(sum(returns))/float(episodes)
        std = np.array(returns).std()
        f = open(os.path.join(self.local_dir, "rewards.csv"), 'a')
        if mean_return > self.best_reward:
            torch.save(self.policy.state_dict(), os.path.join(
                self.model_dir, 'policy.pth'))
            torch.save(self.critic.state_dict(), os.path.join(
                self.model_dir, 'critic.pth'))
            self.best_reward = mean_return
        f.write("%f, %f, %d, %d\n" %
                (mean_return, std, self.steps, self.episodes))
        f.close()

    def __del__(self):
        self.env.close()


class SAC:
    def __init__(self, env, entropy_tuning=True, per=False, n_steps=3000000, local_dir=None):
        self.env = env
        self.entropy_tuning = entropy_tuning
        self.per = per
        self.local_dir = local_dir
        self.num_steps = n_steps
        self.model_dir = os.path.join(local_dir, "model")
        os.makedirs(self.model_dir, exist_ok=True)

        self.batch_size = 256
        self.lr = 1e-4
        self.memory_size = 1000000
        self.gamma = 0.99
        self.ent_coef = 0.2
        self.alpha_mem = 0.6
        self.beta = 0.4
        self.beta_annealing = 0.0001
        self.start_steps = 10000
        self.eval_interval = 10000
        self.log_interval = 10
        self.multi_step = 1

        self.policy = SACActor(self.env.observation_space.shape[0],
                               self.env.action_space.shape[0],
                               hidden_size=256).to(device)
        self.critic = TwinnedQCritic(self.env.observation_space.shape[0],
                                     self.env.action_space.shape[0],
                                     hidden_size=256).to(device)
        self.critic_target = TwinnedQCritic(self.env.observation_space.shape[0],
                                            self.env.action_space.shape[0],
                                            hidden_size=256).to(device).eval()
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False

        self.policy_optimizer = Adam(self.policy.parameters(), lr=self.lr)
        self.q1_optimizer = Adam(self.critic.Q1.parameters(), lr=self.lr)
        self.q2_optimizer = Adam(self.critic.Q2.parameters(), lr=self.lr)

        if self.entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(
                self.env.action_space.shape).to(device)).item()
            self.log_alpha = torch.zeros(
                1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr)
        else:
            self.alpha = torch.tensor(self.ent_coef).to(device)

        if per:
            self.memory = PrioritizedMemory(
                self.memory_size, self.env.observation_space.shape,
                self.env.action_space.shape, device, self.gamma, self.multi_step,
                alpha=self.alpha_mem, beta=self.beta, beta_annealing=self.beta_annealing)
        else:
            self.memory = MultiStepMemory(
                self.memory_size, self.env.observation_space.shape,
                self.env.action_space.shape, device, self.gamma, self.multi_step)

        self.train_rewards = RunningMeanStats(self.log_interval)
        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.best_reward = 0.

    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def is_update(self):
        return len(self.memory) > self.batch_size and self.steps >= self.start_steps

    def act(self, state):
        if self.start_steps > self.steps:
            action = self.env.action_space.sample()
        else:
            action = self.explore(state)
        return action

    def explore(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, _ = self.policy.sample(state)
        return action.cpu().numpy().reshape(-1)

    def exploit(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            _, _, action = self.policy.sample(state)
        return action.cpu().numpy().reshape(-1)

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.critic(states, actions)
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, next_entropies, _ = self.policy.sample(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) + self.alpha * next_entropies
        target_q = rewards + (1.0 - dones) * self.gamma * next_q
        return target_q

    def calc_critic_loss(self, batch, weights):
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)

        errors_q1 = torch.abs(curr_q1.detach() - target_q)
        errors_q2 = torch.abs(curr_q2.detach() - target_q)
        errors = (errors_q1 + errors_q2) / 2

        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)
        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch
        sampled_action, entropy, _ = self.policy.sample(states)
        q1, q2 = self.critic(states, sampled_action)
        q = torch.min(q1, q2)
        policy_loss = torch.mean((-q-self.alpha*entropy)*weights)
        return policy_loss, entropy

    def calc_entropy_loss(self, entropy, weights):
        entropy_loss = - \
            torch.mean(self.log_alpha * (self.target_entropy -
                       entropy).detach() * weights)
        return entropy_loss

    def train_episode(self):
        self.episodes += 1
        episode_reward = 0.
        episode_steps = 0
        done = False
        state = self.env.reset()

        while not done:
            action = self.act(state)
            next_state, reward, done, _ = self.env.step(action)
            self.steps += 1
            episode_steps += 1
            episode_reward += reward

            # ignore done if the agent reach time horizons
            if episode_steps >= self.env._max_episode_steps:
                masked_done = False
            else:
                masked_done = done

            if self.per:
                batch = to_batch(state, action, reward,
                                 next_state, masked_done, device)
                with torch.no_grad():
                    curr_q1, curr_q2 = self.calc_current_q(*batch)
                target_q = self.calc_target_q(*batch)
                error_q1 = torch.abs(curr_q1.detach() - target_q)
                error_q2 = torch.abs(curr_q2.detach() - target_q)
                error = ((error_q1 + error_q2) / 2).item()
                self.memory.append(
                    state, action, reward, next_state, masked_done, error, episode_done=done)
            else:
                self.memory.append(state, action, reward,
                                   next_state, masked_done, episode_done=done)

            if self.is_update():
                self.optimize_model()
            if self.steps % self.eval_interval == 0:
                self.evaluate()

            state = next_state

        self.train_rewards.append(episode_reward)

    def optimize_model(self):
        self.learning_steps += 1
        for t, s in zip(self.critic_target.parameters(), self.critic.parameters()):
            t.data.copy_(t.data * (1.0 - 0.05) + s.data * 0.05)

        if self.per:
            batch, indices, weights = self.memory.sample(self.batch_size)
        else:
            batch = self.memory.sample(self.batch_size)
            weights = 1.

        q1_loss, q2_loss, errors, mean_q1, mean_q2 = self.calc_critic_loss(
            batch, weights)
        update_params(self.q1_optimizer, self.critic.Q1, q1_loss)
        update_params(self.q2_optimizer, self.critic.Q2, q2_loss)

        policy_loss, entropies = self.calc_policy_loss(batch, weights)
        update_params(self.policy_optimizer, self.policy, policy_loss)

        if self.entropy_tuning:
            entropy_loss = self.calc_entropy_loss(entropies, weights)
            update_params(self.alpha_optimizer, None, entropy_loss)
            self.alpha = self.log_alpha.exp()

        if self.per:
            self.memory.update_priority(indices, errors.cpu().numpy())

    def evaluate(self):
        episodes = 10
        returns = np.zeros((episodes,), dtype=np.float32)

        for i in range(episodes):
            state = self.env.reset()
            episode_reward = 0.
            done = False
            while not done:
                action = self.exploit(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
            returns[i] = episode_reward

        mean_return = float(sum(returns)) / float(episodes)
        std = np.array(returns).std()
        f = open(os.path.join(self.local_dir, "rewards.csv"), 'a')
        if mean_return > self.best_reward:
            torch.save(self.policy.state_dict(), os.path.join(
                self.model_dir, 'policy.pth'))
            torch.save(self.critic.state_dict(), os.path.join(
                self.model_dir, 'critic.pth'))
            torch.save(self.critic_target.state_dict(), os.path.join(
                self.model_dir, 'critic_target.pth'))
            self.best_reward = mean_return
        f.write("%f, %f, %d, %d\n" %
                (mean_return, std, self.steps, self.episodes))
        f.close()

    def __del__(self):
        self.env.close()

class RunningMeanStats:
    def __init__(self, n=10):
        self.n = n
        self.stats = deque(maxlen=n)

    def append(self, x):
        self.stats.append(x)

    def get(self):
        return np.mean(self.stats)

class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count