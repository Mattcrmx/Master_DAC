import datetime
import glob
import shutil
from pathlib import Path
from typing import List

import gym
import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch import nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import BatchSampler, SubsetRandomSampler
from memory import Memory
from utils import NothingToDo


class QNet(nn.Module):
    def __init__(self, dim_in, dim_h1, dim_h2, dim_out):
        super().__init__()
        self.m = nn.Sequential(
            nn.Linear(dim_in, dim_h1),
            nn.ReLU(),
            nn.Linear(dim_h1, dim_h2),
            nn.ReLU(),
            nn.Linear(dim_h2, dim_out)
        )

    def forward(self, x):
        return self.m(x)


class RandomAgent:
    """The world's simplest agent!"""

    def __init__(self, env, opt):
        self.opt = opt
        self.env = env
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)

    def act(self, observation, reward, done):
        return self.action_space.sample()

    def save(self, outputDir):
        pass

    def load(self, inputDir):
        pass


class ActorCriticAgent:

    def __init__(self, env, conf):
        self.conf = conf
        self.env = env
        self.action_space = env.action_space
        self.featureExtractor = conf.featExtractor(env)
        self.test = False
        self.transition_buffer = []
        self.steps_v = 0
        self.steps_pi = 0
        self.d_in = self.featureExtractor.outSize
        d_h = conf.dim_h

        # networks init
        self.V_pi = torch.nn.Sequential(
            torch.nn.Linear(self.d_in, 1)
        )
        self.V_pi_tilde = torch.nn.Sequential(
            torch.nn.Linear(self.d_in, 1)
        )
        self.pi = torch.nn.Sequential(
            torch.nn.Linear(self.d_in, d_h),
            torch.nn.Tanh(),
            torch.nn.Linear(d_h, d_h),
            torch.nn.Tanh(),
            torch.nn.Linear(d_h, self.action_space.n),
            torch.nn.Softmax(dim=1)
        )

        # hyper parameters
        self.eps = conf.eps
        self.eps_decay = conf.eps_decay
        self.gamma = conf.gamma
        self.optim_v = torch.optim.Adam(self.V_pi.parameters(), lr=conf.lr)
        self.optim_pi = torch.optim.Adam(self.pi.parameters(), lr=conf.lr)
        self.loss_fct_v = torch.nn.SmoothL1Loss()

    def sampler(self):
        return BatchSampler(
            SubsetRandomSampler(range(len(self.transition_buffer))),
            batch_size=self.conf.batch_size,
            drop_last=True
        )

    def act(self, obs):
        if np.random.uniform() < self.eps:
            self.eps *= self.eps_decay
            # random action selection for exploration
            return self.action_space.sample()
        else:
            return self.get_policy(obs).sample().item()

    def store(self, transition):
        self.transition_buffer.append(transition)

    def train_vpi(self, obs, next_obs, rewards):
        for ep in range(self.conf.epochs_vpi):
            sampler = self.sampler()
            for indices in sampler:
                batch_obs = obs[indices]
                batch_next_obs = next_obs[indices]
                batch_rewards = rewards[indices]

                # Set the gradients to zero
                self.optim_v.zero_grad()

                # forward, backward and optimize
                y_hat = self.V_pi(batch_obs)
                y = batch_rewards.view(-1, 1) + self.gamma * self.V_pi_tilde(batch_next_obs)
                loss = self.loss_fct_v(y_hat, y)
                loss.backward()

                self.optim_v.step()

    def get_policy(self, obs):
        if len(obs.shape) == 1:
            # obs should be a numpy array of size (feature_dim,), make it compatible
            obs = torch.tensor(obs, dtype=torch.float32).view(1, -1)
        logits = self.pi(obs)  # logits tensor
        return Categorical(logits=logits)  # pytorch distribution object

    def learn(self):

        obs = torch.tensor([t[0] for t in self.transition_buffer], dtype=torch.float32)
        actions = torch.tensor([t[1] for t in self.transition_buffer])
        next_obs = torch.tensor([t[2] for t in self.transition_buffer], dtype=torch.float32)
        rewards = torch.tensor([t[3] for t in self.transition_buffer], dtype=torch.float32)

        # fit V_pi to the rewards
        if self.steps_v >= 10000 == 0:
            self.V_pi_tilde.load_state_dict(self.V_pi.state_dict())  # realigns the weights on V_pi
            self.steps_v = 0

        self.train_vpi(obs, next_obs, rewards)

        # evaluate the advantage
        A_pi = rewards + self.gamma * self.V_pi(next_obs)

        # Computes the loss
        logp = self.get_policy(obs).log_prob(actions)
        loss_theta = -(logp * A_pi).mean()

        self.optim_pi.zero_grad()
        loss_theta.backward()
        self.optim_pi.step()  # for policy

        # cleanse the buffer
        self.transition_buffer = []

    def save(self, outputDir):
        pass

    def load(self, inputDir):
        pass

def main():
    PROJECT_DIR = Path(__file__).resolve().parents[0]
    conf = OmegaConf.load(PROJECT_DIR.joinpath('config.yaml'))
    time_tag = datetime.datetime.now().strftime(f'%Y%m%d-%H%M%S')
    log_dir = f'runs/tp4-dqn-{time_tag}-{conf.env}'
    writer = SummaryWriter(log_dir)
    save_src_and_config(log_dir, conf, writer)

    # freqSave = config["freqSave"]
    # nbTest = config["nbTest"]

    # Create a deterministic env
    env = gym.make(conf.env)
    # for gridworld
    # if hasattr(env, 'setPlan'):
    #     env.setPlan(config["map"], config["rewards"])
    seed_everything(conf.seed)
    env.seed(conf.seed)
    env.action_space.seed(conf.seed)

    # agent = RandomAgent(env, config)
    agent = ActorCriticAgent(env, conf)
    memory = Memory(mem_size=conf.mem_size, prior=conf.prior)

    print(f'{log_dir}')
    print(f'Training agent with conf :\n{OmegaConf.to_yaml(conf)}')
    for ep in tqdm(range(conf.episode_count)):
        agent.test = ep % conf.freq_test == 0
        obs = env.reset()
        done = False
        r_sum = 0
        loss = []

        while not done:
            # select action
            action = agent.act(obs, ep)

            # act, observe reward and the transition
            next_obs, reward, done, info = env.step(action)
            reward = reward / 100  # normalize the reward

            if not agent.test and memory.nentities > 2000:
                truncated = info.get("TimeLimit.truncated", False)
                memory.store(transition=(obs, action, next_obs, reward, done, truncated))

                # sample a mini-batch of transitions, and learn
                batch = memory.sample(conf.batch_size)
                l = agent.learn(batch)
                loss.append(l)

                if ep % conf.update_target:
                    agent.update_target_network()

            obs = next_obs.copy()

            r_sum += reward

        if agent.test:
            writer.add_scalar('reward_test', r_sum, ep)
        else:
            writer.add_scalar('reward_train', r_sum, ep)
            writer.add_scalar('loss', np.mean(loss), ep)
        writer.add_scalar('eps', agent.eps, ep)

    env.close()


def save_src_and_config(log_dir: str, conf, writer):
    """
    Save all .py files in the current folder, and the config to log_dir, and log hparams to tensorboard

    Args:
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
