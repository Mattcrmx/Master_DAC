import datetime
import glob
import shutil
from pathlib import Path
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import seed_everything
from torch.distributions import Categorical
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import NothingToDo, MapFromDumpExtractor2, DistsFromStates
import gridworld


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

    def forward(self, x, softmax_dim):
        return F.softmax(self.model(x), dim=softmax_dim)


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


class ActorCritic(nn.Module):
    def __init__(self, action_space, dim_s, gamma, lr, hidden_sizes):
        super(ActorCritic, self).__init__()
        self.data = []
        self.gamma = gamma
        self.lr = lr
        self.pi = PiNet(dim_in=dim_s, hidden_sizes=hidden_sizes, dim_out=action_space.n)
        self.v_pi = VNet(dim_in=dim_s, hidden_sizes=hidden_sizes, dim_out=1)
        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr)
        self.optimizer_v_pi = optim.Adam(self.v_pi.parameters(), lr=self.lr)

    def store(self, transition):
        self.data.append(transition)

    def get_batch(self):
        obs_list, actions_list, rewards_list, obs_next_list, done_lst, trunc_lst = [], [], [], [], [], []
        for transition in self.data:
            obs, action, reward, obs_next, done, info = transition
            obs_list.append(obs)
            actions_list.append([action])
            rewards_list.append([reward / 100.0])
            obs_next_list.append(obs_next)
            done_lst.append([done])
            trunc_lst.append([info.get("TimeLimit.truncated", False)])  # prevent catastrophic forgetting

        self.data = []
        return (torch.tensor(obs_list, dtype=torch.float), torch.tensor(actions_list),
                torch.tensor(rewards_list, dtype=torch.float), torch.tensor(obs_next_list, dtype=torch.float),
                torch.tensor(done_lst, dtype=torch.float), torch.tensor(trunc_lst, dtype=torch.float))

    def learn(self):
        obs, action, reward, obs_next, done, trunc = self.get_batch()
        td_target = reward + self.gamma * self.v_pi(obs_next) * (1 - done + trunc)  # full reward only while not done
        advantage = td_target - self.v_pi(obs)

        pi = self.pi(obs, softmax_dim=1)
        pi_a = pi.gather(1, action)
        loss_pi = -(torch.log(pi_a) * advantage.detach()).mean()
        loss_v_pi = F.smooth_l1_loss(self.v_pi(obs), td_target.detach())

        # train pi
        self.optimizer_pi.zero_grad()
        loss_pi.backward()
        self.optimizer_pi.step()

        # train v_pi
        self.optimizer_v_pi.zero_grad()
        loss_v_pi.backward()
        self.optimizer_v_pi.step()

        return loss_pi.item(), loss_v_pi.item()


def main():
    PROJECT_DIR = Path(__file__).resolve().parents[0]
    conf = OmegaConf.load(PROJECT_DIR.joinpath('config.yaml'))
    time_tag = datetime.datetime.now().strftime(f'%Y%m%d-%H%M%S')
    log_dir = f'runs/tp5-a2c-{time_tag}-{conf.env}-seed{conf.seed}'
    writer = SummaryWriter(log_dir)
    save_src_and_config(log_dir, conf, writer)

    # seed the env
    seed_everything(conf.seed)
    env = gym.make(conf.env)
    env.seed(conf.seed)
    env.action_space.seed(conf.seed)
    print(f'{log_dir}')
    print(f'Training agent with conf :\n{OmegaConf.to_yaml(conf)}')

    if hasattr(env, 'setPlan'):
        # convert OmegaConf object to dict with int keys (not supported by OmegaConf) to match env.setPlan
        reward = OmegaConf.to_container(conf.rewards)
        reward = {int(k): v for k, v in reward.items()}
        env.setPlan(conf.map, reward)

    feat_extractor = None
    if conf.feat_extractor == 'nothing_to_do':
        feat_extractor = NothingToDo(env)
    elif conf.feat_extractor == 'map_from_dump2':
        feat_extractor = MapFromDumpExtractor2(env)
    elif conf.feat_extractor == 'dists_from_states':
        feat_extractor = DistsFromStates(env)

    agent = ActorCritic(gamma=conf.gamma, lr=conf.lr, hidden_sizes=OmegaConf.to_container(conf.hidden_sizes),
                        dim_s=feat_extractor.out_size, action_space=env.action_space)

    for ep in tqdm(range(conf.episode_count)):
        obs = feat_extractor.get_features(env.reset())
        done = False
        r_sum = 0
        while not done:
            for t in range(conf.freq_optim):
                probas = agent.pi(torch.from_numpy(obs).float(), softmax_dim=0)
                multinomial = Categorical(probas)
                action = multinomial.sample().item()
                obs_next, reward, done, info = env.step(action)

                if conf.env != 'gridworld-v0':
                    reward = reward / 100  # normalize the reward

                obs_next = feat_extractor.get_features(obs_next)  # actually use phi(next_obs)
                agent.store((obs, action, reward, obs_next, done, info))

                obs = obs_next.copy()
                r_sum += reward

                if done:
                    break

            # Learn every freq_optim time-steps or when the episodes ends (whichever comes earliest)
            loss_pi, loss_v_pi = agent.learn()
            writer.add_scalar('loss_pi', loss_pi, ep)
            writer.add_scalar('loss_vpi', loss_v_pi, ep)
        writer.add_scalar('reward_train', r_sum, ep)

    env.close()


def save_src_and_config(log_dir: str, conf, writer):
    # save config and source files as text files
    with open(f'{log_dir}/conf.yaml', 'w') as f:
        OmegaConf.save(conf, f)
    for f in glob.iglob('*.py'):
        shutil.copy2(f, log_dir)

    conf_clean = {k: str(v) for (k, v) in conf.items()}
    writer.add_hparams(conf_clean, metric_dict={'score': 0.})


if __name__ == '__main__':
    main()
