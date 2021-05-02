import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import gym
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch import nn, optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from utils import MapFromDumpExtractor2, NothingToDo, DistsFromStates, save_src_and_config
import gridworld  # to register the env


class Memory:
    def __init__(self):
        self.memory = []

    def store(self, transition):
        self.memory.append(transition)

    def get_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst, trunc_lst = [], [], [], [], [], []

        for transition in self.memory:
            s, a, r, s_prime, done, info = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_lst.append([done])
            trunc_lst.append([info.get("TimeLimit.truncated", False)])  # True if the episode was truncated (time limit)
        self.memory = []

        N, dim_a = len(a_lst), len(a_lst[0])
        return (torch.tensor(s_lst, dtype=torch.float),  # (N, dim_s)
                torch.tensor(a_lst).reshape(N, dim_a),  # (N, dim_a) even if dim_a=1 thx to reshape
                torch.tensor(r_lst, dtype=torch.float),  # (N, 1)
                torch.tensor(s_prime_lst, dtype=torch.float),  # (N, dim_s)
                torch.tensor(done_lst, dtype=torch.float),  # (N, 1)
                torch.tensor(trunc_lst, dtype=torch.float))  # (N, 1)


class PiNet(nn.Module):
    def __init__(self, dim_in, hidden_sizes, dim_out):
        super().__init__()
        sizes = [dim_in] + hidden_sizes
        layers = []

        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]),
                       nn.ReLU()]
        layers += [nn.Linear(sizes[-1], dim_out)]  # last layer

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class VNet(nn.Module):
    def __init__(self, dim_in, hidden_sizes, dim_out):
        super().__init__()
        sizes = [dim_in] + hidden_sizes
        layers = []

        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]),
                       nn.ReLU()]
        layers += [nn.Linear(sizes[-1], dim_out)]  # last layer

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PPO:
    def __init__(self, dim_s, action_space, hidden_sizes, lr, gamma, K, clip_eps, delta=None):
        self.gamma = gamma
        self.K = K
        self.clip_eps = clip_eps
        self.delta = delta
        self.pi = PiNet(dim_in=dim_s, hidden_sizes=hidden_sizes, dim_out=action_space.n)
        self.v = VNet(dim_in=dim_s, hidden_sizes=hidden_sizes, dim_out=1)
        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=lr)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=lr)

    def act(self, obs):
        logits = self.pi(torch.from_numpy(obs).float())
        multinomial = Categorical(logits=logits)
        return multinomial.sample().item()

    def learn(self, batch, beta_k):
        obs, action, reward, obs_next, done, trunc = batch
        not_finished = (1 - done + trunc)  # full reward only while not done

        # probas of the policy used to obtain the samples
        pi_k_logits = self.pi(obs).detach()
        pi_k = F.softmax(pi_k_logits, dim=1)
        pi_k_a = pi_k.gather(1, action)
        loss_pi_log = []
        loss_v_log = []
        loss_adv_log = []

        # todo: TD(lambda) for advantage estimation and value estimation
        td_target = (reward + self.gamma * self.v(obs_next) * not_finished).detach()
        advantage = (td_target - self.v(obs)).detach()
        for s in range(self.K):
            # train pi
            pi_logits = self.pi(obs)
            pi = F.softmax(pi_logits, dim=1)
            pi_a = pi.gather(1, action)

            ratio = pi_a / pi_k_a

            if self.clip_eps > 0:
                # clipped version
                loss_1 = (ratio * advantage)
                loss_2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantage
                loss_pi = -torch.min(loss_1, loss_2).mean()
            else:
                # KL div version
                loss_pi = -(ratio * advantage).mean() + beta_k * F.kl_div(F.log_softmax(pi_k_logits, dim=1),
                                                                          F.log_softmax(pi_logits, dim=1),
                                                                          log_target=True,
                                                                          reduction='batchmean')

            self.optimizer_pi.zero_grad()
            loss_pi.backward()
            self.optimizer_pi.step()

            # train v_pi
            loss_v = F.smooth_l1_loss(self.v(obs), td_target)
            self.optimizer_v.zero_grad()
            loss_v.backward()
            self.optimizer_v.step()

            loss_adv_log.append((ratio * advantage).mean().item())
            loss_pi_log.append(loss_pi.item())
            loss_v_log.append(loss_v.item())

        if self.clip_eps < 0:  # KL div version
            kl_div = F.kl_div(F.log_softmax(pi_k_logits, dim=1), F.log_softmax(self.pi(obs), dim=1), log_target=True,
                              reduction='batchmean')

            if kl_div > 1.5 * self.delta:
                beta_k *= 2
            elif kl_div <= self.delta / 1.5:
                beta_k *= 0.5

            return np.mean(loss_pi_log), np.mean(loss_v_log), beta_k, kl_div, np.mean(loss_adv_log)

        else:
            return np.mean(loss_pi_log), np.mean(loss_v_log)


def main():
    PROJECT_DIR = Path(__file__).resolve().parents[0]
    conf = OmegaConf.load(PROJECT_DIR.joinpath('config.yaml'))
    time_tag = datetime.datetime.now().strftime(f'%Y%m%d-%H%M%S')
    ppl_algo = 'kldiv' if conf.clip_eps < 0 else f'clip_eps{conf.clip_eps}'
    log_dir = f'runs/tp6-{time_tag}-{conf.env}-{ppl_algo}-delta{conf.delta}-seed{conf.seed}'
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

    memory = Memory()
    agent = PPO(dim_s=feat_extractor.out_size, action_space=env.action_space,
                hidden_sizes=OmegaConf.to_container(conf.hidden_sizes), lr=conf.lr, gamma=conf.gamma, K=conf.K,
                clip_eps=conf.clip_eps, delta=conf.delta)

    print(f'{log_dir}')
    print(f'Training agent with conf :\n{OmegaConf.to_yaml(conf)}')
    beta_k = 1
    for ep in tqdm(range(conf.episode_count)):
        obs = feat_extractor.get_features(env.reset())
        done = False
        r_sum = 0
        while not done:
            # act and observe the next step
            act = agent.act(obs)
            obs_next, reward, done, info = env.step(act)

            # save it
            obs_next = feat_extractor.get_features(obs_next)  # actually use phi(next_obs)
            if conf.env != 'gridworld-v0':
                reward = reward / 100  # normalize the reward
            memory.store(transition=(obs, act, reward, obs_next, done, info))

            obs = obs_next.copy()
            r_sum += reward

        if agent.clip_eps > 0:
            loss_pi, loss_v_pi = agent.learn(memory.get_batch(), beta_k=beta_k)
        else:
            loss_pi, loss_v_pi, beta_k, kl_div, loss_adv = agent.learn(memory.get_batch(), beta_k=beta_k)
            writer.add_scalar('beta', beta_k, ep)
            writer.add_scalar('kl', kl_div, ep)
            writer.add_scalar('loss_adv', loss_adv, ep)

        writer.add_scalar('loss_pi', loss_pi, ep)
        writer.add_scalar('loss_vpi', loss_v_pi, ep)
        writer.add_scalar('reward', r_sum, ep)

    env.close()


if __name__ == '__main__':
    main()
