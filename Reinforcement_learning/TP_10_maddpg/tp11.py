import datetime
from pathlib import Path
from time import sleep
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
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

logging.basicConfig(level=logging.INFO)


class QNet(nn.Module):
    def __init__(self, dim_s: List[int], dim_a: List[int], hidden_sizes: List[int]):
        super(QNet, self).__init__()
        sizes = [sum(dim_s) + sum(dim_a)] + hidden_sizes
        layers = []
        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]),
                       nn.LeakyReLU()]
        layers += [nn.Linear(sizes[-1], 1)]  # last layer
        self.model = nn.Sequential(*layers)

    def forward(self, s: List[torch.tensor], a: List[torch.tensor]):
        # a: n_agents * (N, dim_a_i)
        # s: n_agents * (N, dim_s_i)
        a_cat = torch.cat(a, dim=1)
        s_cat = torch.cat(s, dim=1)
        return self.model(torch.cat([s_cat, a_cat], dim=1))


class MuNet(nn.Module):
    def __init__(self, dim_s: int, dim_a_i: int, hidden_sizes: List[int]):
        super(MuNet, self).__init__()
        sizes = [dim_s] + hidden_sizes
        layers = []
        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]),  # hidden layers
                       nn.LeakyReLU()]
        layers += [nn.Linear(sizes[-1], dim_a_i),  # last layer
                   nn.Tanh()]  # tanh to normalize output
        self.model = nn.Sequential(*layers)

    def forward(self, s):
        return self.model(s)


class ReplayBuffer:
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, s: List[np.array], a: List[np.array], r: List[float], s_prime: List[np.array], done: List[bool],
            info: dict):
        self.buffer.append((s, a, r, s_prime, done, info))

    def sample(self, N: int):
        mini_batch = random.sample(self.buffer, N)
        n_agents = len(mini_batch[0][0])  # number of states in the first transition of the mini-batch
        s_lst, a_lst, r_lst, s_prime_lst, done_lst, trunc_lst = (
            [[] for i in range(n_agents)], [[] for i in range(n_agents)], [[] for i in range(n_agents)],
            [[] for i in range(n_agents)], [[] for i in range(n_agents)], [[] for i in range(n_agents)]
        )

        for transition in mini_batch:
            s, a, r, s_prime, done, info = transition
            for i in range(n_agents):
                s_lst[i].append(s[i])
                a_lst[i].append(a[i])
                r_lst[i].append([r[i]])
                s_prime_lst[i].append(s_prime[i])
                done_lst[i].append([done[i]])
                # True if the episode was truncated (time limit)
                trunc_lst[i].append([info['n'][i].get("TimeLimit.truncated", False)])

        return ([torch.tensor(s_lst[i], dtype=torch.float) for i in range(n_agents)],  # n_agents * (N, dim_s)
                [torch.tensor(a_lst[i], dtype=torch.float) for i in range(n_agents)],  # n_agents * (N, dim_a)
                [torch.tensor(r_lst[i], dtype=torch.float) for i in range(n_agents)],  # n_agents * (N, 1)
                [torch.tensor(s_prime_lst[i], dtype=torch.float) for i in range(n_agents)],  # n_agents * (N, dim_s)
                [torch.tensor(done_lst[i], dtype=torch.float) for i in range(n_agents)],  # n_agents * (N, 1)
                [torch.tensor(trunc_lst[i], dtype=torch.float) for i in range(n_agents)])  # n_agents * (N, 1)

    def __len__(self):
        return len(self.buffer)


class DDPG:
    def __init__(self, dim_s: List[int], dim_a: List[int], hidden_sizes_q: List[int], hidden_sizes_mu: List[int],
                 gamma: float, lr_q: float, lr_mu: float, rho: float, eps_std: float, agent_index: int):
        self.gamma = gamma
        self.rho = rho
        self.agent_index = agent_index
        self.eps_std = eps_std
        self.q_current = QNet(dim_s, dim_a, hidden_sizes_q)
        self.q_target = QNet(dim_s, dim_a, hidden_sizes_q)
        self.mu_current = MuNet(dim_s[agent_index], dim_a[agent_index], hidden_sizes_mu)
        self.mu_target = MuNet(dim_s[agent_index], dim_a[agent_index], hidden_sizes_mu)
        self.optim_q = optim.Adam(params=self.q_current.parameters(), lr=lr_q)
        self.optim_mu = optim.Adam(params=self.mu_current.parameters(), lr=lr_mu)

        # initialize the weights of the target networks with the current networks
        self.q_target.load_state_dict(self.q_current.state_dict())
        self.mu_target.load_state_dict(self.mu_current.state_dict())

    def act(self, s: torch.tensor):
        with torch.no_grad():
            a = self.mu_current(s)
        eps = torch.normal(mean=0, std=self.eps_std, size=a.shape)
        return torch.clamp(a + eps, min=-1., max=1.)

    def learn(self, batch, agents):
        s, a, r, s_prime, done, is_truncated = batch
        s_i = s[self.agent_index]
        r_i = r[self.agent_index]
        done_i = done[self.agent_index]
        is_truncated_i = is_truncated[self.agent_index]

        # gradient descent on Q
        with torch.no_grad():
            # 1 if the episode is not done yet, or if it's done but it has been truncated because of a time limit
            not_finished = (1 - done_i + is_truncated_i)
            a_prime = [agent.mu_target(s_prime[i]) for (i, agent) in enumerate(agents)]
            targets = r_i + self.gamma * not_finished * self.q_target(s_prime, a_prime)
        loss_q = F.smooth_l1_loss(self.q_current(s, a), targets.detach())
        self.optim_q.zero_grad()
        loss_q.backward()
        self.optim_q.step()

        # gradient ascent on Mu
        a[self.agent_index] = self.mu_current(s_i)
        loss_mu = - self.q_current(s, a).mean()
        self.optim_mu.zero_grad()
        loss_mu.backward()
        self.optim_mu.step()

        return loss_q, loss_mu

    def update_target_networks(self):
        for p_target, p_current in zip(self.q_target.parameters(), self.q_current.parameters()):
            p_target.data.copy_(p_target.data * self.rho + p_current.data * (1.0 - self.rho))
        for p_target, p_current in zip(self.mu_target.parameters(), self.mu_current.parameters()):
            p_target.data.copy_(p_target.data * self.rho + p_current.data * (1.0 - self.rho))


def make_env(scenario_name, benchmark=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    world.dim_c = 0
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    env.discrete_action_space = False
    env.discrete_action_input = False
    scenario.reset_world(world)
    return env, scenario, world


def main():
    PROJECT_DIR = Path(__file__).resolve().parents[0]
    conf = OmegaConf.load(PROJECT_DIR.joinpath('conf.yaml'))
    time_tag = datetime.datetime.now().strftime(f'%Y%m%d-%H%M%S')
    writer = SummaryWriter(f'runs/MADDPG-{time_tag}-{conf.env_name}')

    # Logging hyper params
    print(f'Running with conf: \n{OmegaConf.to_yaml(conf)}')
    conf_clean = {k: str(v) for (k, v) in conf.items()}
    writer.add_hparams(conf_clean, metric_dict={'score': 0})

    # environment loading
    env, scenario, world = make_env(conf.env_name)
    # seed everything to get reproducible runs
    seed_everything(conf.seed)
    env.seed(conf.seed)

    # for each agent
    agents = []
    dim_a = [2] * env.n  # force applied on x, y axis (in [-1, 1])
    first_obs = env.reset()
    dim_s = [len(obs_i) for obs_i in first_obs]
    for i in range(env.n):
        env.action_space[i].seed(conf.seed)
        agent = DDPG(dim_s=dim_s, dim_a=dim_a, hidden_sizes_q=list(conf.hidden_sizes_q),
                     hidden_sizes_mu=list(conf.hidden_sizes_mu), gamma=conf.gamma, lr_q=conf.lr_q, lr_mu=conf.lr_mu,
                     rho=conf.rho, eps_std=conf.eps_std, agent_index=i)
        agents.append(agent)

    buffer = ReplayBuffer(conf.buffer_limit)

    for episode in tqdm(range(conf.max_episodes)):
        # training loop
        obs = env.reset()
        done = False
        r_sum = [0.] * env.n
        t = 0
        while not done and t < 25:
            # act and observe the result
            actions = [agent.act(torch.tensor(obs[i], dtype=torch.float)).detach().numpy() for (i, agent) in
                       enumerate(agents)]
            obs_next, rewards, done_list, info = env.step(actions)
            done = np.any(done_list)

            # store transition
            # rewards = [r / 100. for r in rewards]  # normalize the reward
            buffer.put(obs, actions, rewards, obs_next, done_list, info)
            obs = obs_next.copy()
            r_sum = [r_sum[i] + rewards[i] for i in range(env.n)]

            t += 1
            if conf.render and episode % 100 == 0:
                print(f'ep {episode} - rendering t={t}')
                env.render(mode='none')

        # learning step
        if len(buffer) >= conf.buffer_min_len and episode % conf.update_freq == 0:
            for i, agent in enumerate(agents):
                loss_q, loss_mu = [], []
                for _ in range(conf.update_iter):
                    l_q, l_mu = agent.learn(buffer.sample(conf.batch_size), agents)
                    loss_q += [l_q.item()]
                    loss_mu += [l_mu.item()]
                writer.add_scalar(f'loss_q_{i}', np.mean(loss_q), episode)
                writer.add_scalar(f'loss_mu_{i}', np.mean(loss_mu), episode)

            for agent in agents:
                agent.update_target_networks()

        # log rewards
        for i in range(env.n):
            writer.add_scalar(f'reward_{i}', r_sum[i], episode)


if __name__ == '__main__':
    main()
