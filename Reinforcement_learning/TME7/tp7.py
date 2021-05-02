import datetime
from pathlib import Path
from typing import List

from tqdm import tqdm
import gym
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import OmegaConf
import logging

from pytorch_lightning import seed_everything
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO)


class QNet(nn.Module):
    def __init__(self, dim_s: int, dim_a: int, hidden_sizes: List[int]):
        super(QNet, self).__init__()
        sizes = [dim_s + dim_a] + hidden_sizes
        layers = []
        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]),
                       nn.ReLU()]
        layers += [nn.Linear(sizes[-1], 1)]  # last layer
        self.model = nn.Sequential(*layers)

    def forward(self, s, a):
        # a: (N, dim_a) s: (N, dim_s)
        return self.model(torch.cat([s, a], dim=1))


class MuNet(nn.Module):
    def __init__(self, dim_s: int, dim_a: int, hidden_sizes: List[int], action_bound: float):
        super(MuNet, self).__init__()
        self.action_bound = action_bound

        sizes = [dim_s] + hidden_sizes
        layers = []
        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]),  # hidden layers
                       nn.ReLU()]
        layers += [nn.Linear(sizes[-1], dim_a),  # last layer
                   nn.Tanh()]  # tanh to normalize output
        self.model = nn.Sequential(*layers)

    def forward(self, s):
        return self.action_bound * self.model(s)


class ReplayBuffer:
    # Inspired from https://github.com/seungeunrho/minimalRL/blob/master/ddpg.py
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_lst, trunc_lst = [], [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done, info = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_lst.append([done])
            trunc_lst.append([info.get("TimeLimit.truncated", False)])  # True if the episode was truncated (time limit)

        N, dim_a = len(mini_batch), len(a_lst[0])
        return (torch.tensor(s_lst, dtype=torch.float),  # (N, dim_s)
                torch.tensor(a_lst, dtype=torch.float).reshape(N, dim_a),  # (N, dim_a) even if dim_a=1 thx to reshape
                torch.tensor(r_lst, dtype=torch.float),  # (N, 1)
                torch.tensor(s_prime_lst, dtype=torch.float),  # (N, dim_s)
                torch.tensor(done_lst, dtype=torch.float),  # (N, 1)
                torch.tensor(trunc_lst, dtype=torch.float))  # (N, 1)

    def size(self):
        return len(self.buffer)


class DDPG:
    def __init__(self, dim_s: int, dim_a: int, hidden_sizes_q: List[int], hidden_sizes_mu: List[int], a_low: float,
                 a_high: float, gamma: float, lr_q: float, lr_mu: float, rho: float, eps_std: float,
                 action_bound: float):
        self.a_low = a_low
        self.a_high = a_high
        self.gamma = gamma
        self.rho = rho
        self.eps_std = eps_std
        self.q_current = QNet(dim_s, dim_a, hidden_sizes_q)
        self.q_target = QNet(dim_s, dim_a, hidden_sizes_q)
        self.mu_current = MuNet(dim_s, dim_a, hidden_sizes_mu, action_bound=action_bound)
        self.mu_target = MuNet(dim_s, dim_a, hidden_sizes_mu, action_bound=action_bound)
        self.optim_q = optim.Adam(params=self.q_current.parameters(), lr=lr_q)
        self.optim_mu = optim.Adam(params=self.mu_current.parameters(), lr=lr_mu)

        # initialize the weights of the target networks with the current networks
        self.q_target.load_state_dict(self.q_current.state_dict())
        self.mu_target.load_state_dict(self.mu_current.state_dict())

    def act(self, s):
        with torch.no_grad():
            a = self.mu_current(s)
        eps = torch.normal(mean=0, std=self.eps_std, size=a.shape)
        return torch.clamp(a + eps, min=self.a_low, max=self.a_high)

    def learn(self, batch):
        s, a, r, s_prime, done, is_truncated = batch

        # gradient descent on Q
        with torch.no_grad():
            # 1 if the episode is not done yet, or if it's done but it has been truncated because of a time limit
            not_finished = (1 - done + is_truncated)
            targets = r + self.gamma * not_finished * self.q_target(s_prime, self.mu_target(s_prime))
        loss_q = F.smooth_l1_loss(self.q_current(s, a), targets.detach())
        self.optim_q.zero_grad()
        loss_q.backward()
        self.optim_q.step()

        # gradient ascent on Mu
        loss_mu = - self.q_current(s, self.mu_current(s)).mean()
        self.optim_mu.zero_grad()
        loss_mu.backward()
        self.optim_mu.step()

        # update target networks
        for p_target, p_current in zip(self.q_target.parameters(), self.q_current.parameters()):
            p_target.data.copy_(p_target.data * self.rho + p_current.data * (1.0 - self.rho))
        for p_target, p_current in zip(self.mu_target.parameters(), self.mu_current.parameters()):
            p_target.data.copy_(p_target.data * self.rho + p_current.data * (1.0 - self.rho))

        return loss_q, loss_mu


def main():
    PROJECT_DIR = Path(__file__).resolve().parents[0]
    conf = OmegaConf.load(PROJECT_DIR.joinpath('config.yaml'))
    time_tag = datetime.datetime.now().strftime(f'DDPG-%Y%m%d-%H%M%S-{conf.env_name}')
    writer = SummaryWriter(f'runs/{time_tag}')

    # environment loading
    env = gym.make(conf.env_name)
    action_bound = max(abs(env.action_space.high.max()), abs(env.action_space.low.min()))
    logging.info(f'action bound: {action_bound}')

    # seed everything to get reproducible runs
    seed_everything(conf.seed)
    env.seed(conf.seed)
    env.action_space.seed(conf.seed)

    dim_s, = env.observation_space.shape
    dim_a, = env.action_space.shape
    agent = DDPG(dim_s=dim_s, dim_a=dim_a, hidden_sizes_q=list(conf.hidden_sizes_q),
                 hidden_sizes_mu=list(conf.hidden_sizes_mu), a_low=conf.a_low, a_high=conf.a_high, gamma=conf.gamma,
                 lr_q=conf.lr_q, lr_mu=conf.lr_mu, rho=conf.rho, eps_std=conf.eps_std, action_bound=action_bound)
    buffer = ReplayBuffer(conf.buffer_limit)

    score = 0
    last_rewards = []
    for episode in tqdm(range(conf.max_episodes)):
        # training loop
        obs = env.reset()
        done = False
        r_sum = 0
        while not done:
            # act and observe the result
            action = agent.act(torch.tensor(obs, dtype=torch.float)).detach().numpy()
            obs_next, reward, done, info = env.step(action)

            # store transition
            reward = reward / 100.  # normalize the reward
            buffer.put(transition=(obs, action, reward, obs_next, done, info))
            obs = obs_next.copy()
            r_sum += reward

        # learning step
        if buffer.size() >= 2000 and episode % conf.update_freq == 0:
            loss_q, loss_mu = [], []
            for _ in range(conf.update_iter):
                l_q, l_mu = agent.learn(buffer.sample(conf.batch_size))
                loss_q += [l_q.item()]
                loss_mu += [l_mu.item()]
            writer.add_scalar('loss_q', np.mean(loss_q), episode)
            writer.add_scalar('loss_mu', np.mean(loss_mu), episode)

        # log rewards
        score += r_sum
        last_rewards.append(r_sum)
        writer.add_scalar('reward', r_sum, episode)
        writer.add_scalar('cum_reward', score, episode)
        if len(last_rewards) == 10:
            writer.add_scalar('avg_reward', np.mean(last_rewards), episode)
            last_rewards = []

    # Logging
    conf_clean = {k: str(v) for (k, v) in conf.items()}
    writer.add_hparams(conf_clean, metric_dict={'score': score})


if __name__ == '__main__':
    main()
