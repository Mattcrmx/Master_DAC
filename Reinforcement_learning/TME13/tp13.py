import datetime
import glob
import shutil
from pathlib import Path

import gym
import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import gridworld
from memory import Memory
from utils import NothingToDo, MapFromDumpExtractor2, DistsFromStates, MapFromDumpExtractor4


class QNet(nn.Module):
    def __init__(self, dim_in, hidden_sizes, dim_out):
        super().__init__()
        sizes = [dim_in] + hidden_sizes
        layers = []

        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]),
                       nn.Tanh()]
        layers += [nn.Linear(sizes[-1], dim_out)]  # last layer

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DQNAgent:
    def __init__(self, action_space, dim_s_prime, eps, eps_decay, gamma, h_sizes, lr):
        self.test = False
        self.action_space = action_space
        self.eps = eps
        self.eps_decay = eps_decay
        self.gamma = gamma

        # instantiate our Q networks (current and target)
        self.q = QNet(dim_in=dim_s_prime, hidden_sizes=h_sizes, dim_out=self.action_space.n)
        self.q_target = QNet(dim_in=dim_s_prime, hidden_sizes=h_sizes, dim_out=self.action_space.n)
        self.q_target.load_state_dict(self.q.state_dict())

        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)

    def act(self, obs, goal):
        if np.random.uniform() < self.eps and (not self.test):
            # random action selection for exploration
            return self.action_space.sample()
        else:
            # greedy action selection on optimal Q value estimate
            with torch.no_grad():
                obs = torch.tensor(obs, dtype=torch.float32)
                goal = torch.tensor(goal, dtype=torch.float32)
                obs_goal = torch.cat([obs, goal], dim=0).view(1, -1)
                a = torch.argmax(self.q(obs_goal)).item()
                return a

    def learn(self, batch):
        # decay at each episode
        self.eps *= self.eps_decay

        # compute loss
        obs = torch.tensor([t[0] for t in batch], dtype=torch.float)  # (N, 2)
        act = torch.tensor([t[1] for t in batch]).unsqueeze(1)  # (N, 1)
        next_obs = torch.tensor([t[2] for t in batch], dtype=torch.float)  # (N, 2)
        reward = torch.tensor([t[3] for t in batch], dtype=torch.float).unsqueeze(1)  # (N, 1)
        done = torch.tensor([t[4] for t in batch], dtype=torch.float).unsqueeze(1)  # (N, 1)
        goal = torch.tensor([t[5] for t in batch], dtype=torch.float)  # (N, 2)

        obs_goal = torch.cat([obs, goal], dim=1)  # (N, 4)
        next_obs_goal = torch.cat([next_obs, goal], dim=1)  # (N, 4)

        with torch.no_grad():
            q_next, _ = self.q_target(next_obs_goal).max(dim=1)  # (N,)
            q_target = reward + self.gamma * q_next.unsqueeze(1) * (1 - done)  # (N, 1)

        q_pred = self.q(obs_goal).gather(1, act)

        loss = nn.functional.smooth_l1_loss(q_pred, q_target.detach())

        # optimize parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.q_target.load_state_dict(self.q.state_dict())

    def save(self, outputDir):
        pass

    def load(self, inputDir):
        pass


def main():
    PROJECT_DIR = Path(__file__).resolve().parents[0]
    conf = OmegaConf.load(PROJECT_DIR.joinpath('config.yaml'))
    time_tag = datetime.datetime.now().strftime(f'%Y%m%d-%H%M%S')
    log_dir = f'runs/tp13-dqngoal-{time_tag}-{conf.env}-seed{conf.seed}-eps{conf.eps}'
    writer = SummaryWriter(log_dir)
    save_src_and_config(log_dir, conf, writer)

    # Create a deterministic env
    seed_everything(conf.seed)
    env = gym.make(conf.env)
    env.seed(conf.seed)
    env.action_space.seed(conf.seed)
    # for gridworld
    if hasattr(env, 'setPlan'):
        # convert OmegaConf object to dict with int keys (not supported by OmegaConf) to match env.setPlan
        rewards = OmegaConf.to_container(conf.rewards)
        rewards = {int(k): v for k, v in rewards.items()}
        env.setPlan(conf.map, rewards)

    feat_extractor = None
    if conf.feat_extractor == 'nothing_to_do':
        feat_extractor = NothingToDo(env)
    elif conf.feat_extractor == 'map_from_dump2':
        feat_extractor = MapFromDumpExtractor2(env)
    elif conf.feat_extractor == 'dists_from_states':
        feat_extractor = DistsFromStates(env)
    elif conf.feat_extractor == 'map_from_dump_4':
        feat_extractor = MapFromDumpExtractor4(env)

    agent = DQNAgent(action_space=env.action_space, dim_s_prime=2 * feat_extractor.outSize, eps=conf.eps,
                     eps_decay=conf.eps_decay, gamma=conf.gamma,
                     h_sizes=OmegaConf.to_container(conf.hidden_sizes), lr=conf.lr)
    memory = Memory(mem_size=conf.mem_size, prior=conf.prior)

    print(f'{log_dir}')
    print(f'Training agent with conf :\n{OmegaConf.to_yaml(conf)}')
    for ep in tqdm(range(conf.episode_count)):
        agent.test = ep % conf.freq_test == 0
        obs = feat_extractor.getFeatures(env.reset()).squeeze()
        done = False
        r_sum = 0
        goal, _ = env.sampleGoal()
        goal = feat_extractor.getFeatures(goal).squeeze()
        ep_length = 0

        while ep_length < conf.max_ep_len:
            # sample goal

            # act, observe reward and the transition
            action = agent.act(obs, goal)
            next_obs, _, _, _ = env.step(action)
            next_obs = feat_extractor.getFeatures(next_obs).squeeze()  # actually use phi(next_obs)
            done = (next_obs == goal).all()
            reward = 1 if done else -0.1
            memory.store(transition=(obs, action, next_obs, reward, done, goal))

            obs = next_obs.copy()

            r_sum += reward
            ep_length += 1

        if not agent.test and len(memory) > 2000:
            # sample a mini-batch of transitions, and learn
            batch = memory.sample(conf.batch_size)
            loss = agent.learn(batch)
            writer.add_scalar('loss', loss, ep)

        if ep % conf.update_target:
            agent.update_target_network()

        if agent.test:
            writer.add_scalar('reward_test', r_sum, ep)
        else:
            writer.add_scalar('reward_train', r_sum, ep)
        writer.add_scalar('buffer_size', len(memory), ep)
        writer.add_scalar('eps', agent.eps, ep)

    env.close()


def save_src_and_config(log_dir: str, conf, writer):
    """
    Save all .py files in the current folder, and the config to log_dir, and log hparams to tensorboard

    Args:
        writer:
        log_dir:
        conf:

    Returns:

    """
    # save config and source files as text files
    with open(f'{log_dir}/conf.yaml', 'w') as f:
        OmegaConf.save(conf, f)
    for f in glob.iglob('*.py'):
        shutil.copy2(f, log_dir)

    conf_clean = {k: str(v) for (k, v) in conf.items()}
    writer.add_hparams(conf_clean, metric_dict={'score': 0.})


if __name__ == '__main__':
    main()
