import datetime
import glob
import pickle
import shutil
from pathlib import Path
import numpy as np
import gym
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F


class Memory:
    def __init__(self, batch_size):
        self.memory = []
        self.cumulated_r = []
        self.batch_size = batch_size

    def store(self, transition):
        self.memory.append(transition)

    def get_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        indices = range(len(self.memory))
        for i in indices:
            s, a, r, s_prime, done, info = self.memory[i]
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_lst.append([done])

        N, dim_a = len(a_lst), len(a_lst[0])
        return (torch.tensor(s_lst, dtype=torch.float),  # (N, dim_s)
                torch.tensor(a_lst).reshape(N, dim_a),  # (N, dim_a) even if dim_a=1 thx to reshape
                torch.tensor(r_lst, dtype=torch.float),  # (N, 1)
                torch.tensor(s_prime_lst, dtype=torch.float),  # (N, dim_s)
                torch.tensor(done_lst, dtype=torch.float))  # (N, 1)

    def get_minibatch_proxy_reward(self):
        assert len(self.cumulated_r) == len(self.memory)
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []

        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        for i in indices:
            s, a, r, s_prime, done, info = self.memory[i]
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([self.cumulated_r[i]])
            s_prime_lst.append(s_prime)
            done_lst.append([done])

        N, dim_a = len(a_lst), len(a_lst[0])
        return (torch.tensor(s_lst, dtype=torch.float),  # (N, dim_s)
                torch.tensor(a_lst).reshape(N, dim_a),  # (N, dim_a) even if dim_a=1 thx to reshape
                torch.tensor(r_lst, dtype=torch.float),  # (N, 1)
                torch.tensor(s_prime_lst, dtype=torch.float),  # (N, dim_s)
                torch.tensor(done_lst, dtype=torch.float))  # (N, 1)

    def compute_cumulated_r(self, agent, dim_a):
        cumulated_rewards = []
        cumulated_rewards_real = []
        r_cumulated = 0
        r_cumulated_real = 0
        remaining_ep_len = 0

        # get GAIL rewards (using the discriminator) instead of real rewards
        obs, act, _, _, _ = self.get_batch()
        act_one_hot = F.one_hot(act.flatten(), num_classes=dim_a)
        input_agent = torch.cat([obs, act_one_hot], dim=1)
        d_agent = torch.sigmoid(agent.discriminator(input_agent))
        rewards_clipped = torch.clamp(torch.log(d_agent), min=-100, max=0)
        rewards_clipped = torch.flatten(rewards_clipped).tolist()

        for i in reversed(range(len(self.memory))):
            s, a, r, s_prime, done, info = self.memory[i]
            if done:
                remaining_ep_len = 0
                r_cumulated = 0
                r_cumulated_real = 0
            r_cumulated_real = r + r_cumulated_real  # R_t = sum_{t'=t}^T r_t'
            r_cumulated = rewards_clipped[i] + r_cumulated  # R_t = sum_{t'=t}^T r_t'
            remaining_ep_len += 1  # T - t
            cumulated_rewards.append(r_cumulated / remaining_ep_len)
            cumulated_rewards_real.append(r_cumulated_real / remaining_ep_len)
        self.cumulated_r = list(reversed(cumulated_rewards))

    def __len__(self):
        return len(self.memory)


class ExpertAgent:
    def __init__(self, dim_a, dim_s, file):
        self.dim_a = dim_a
        self.dim_s = dim_s
        with open(file, 'rb') as handle:
            self.expert_data = pickle.load(handle)  # FloatTensor
            self.states = self.expert_data[:, : self.dim_s]
            self.actions = self.expert_data[:, self.dim_s:]
            self.states = self.states.contiguous()
            self.actions = self.actions.contiguous()
            self.actions = self.actions.argmax(dim=1)  # index instead of one-hot

    # def to_one_hot(self, actions):
    #     actions = actions.view(-1)  # LongTensor
    #     one_hot = torch.zeros(actions.size()[0], self.dim_a)  # FloatTensor
    #     one_hot[range(actions.size()[0]), actions] = 1
    #     return one_hot
    #
    # def to_index_action(self, one_hot):
    #     ac = self.longTensor.new(range(self.dim_a)).view(1, -1)
    #     ac = ac.expand(one_hot.size()[0], -1).contiguous().view(-1)
    #     actions = ac[one_hot.view(-1) > 0].view(-1)
    #     return actions


class BehavioralCloning:
    def __init__(self, dim_s, dim_a, lr):
        self.pi = nn.Sequential(
            nn.Linear(dim_s, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, dim_a)
        )
        self.optim = torch.optim.Adam(self.pi.parameters(), lr=lr)

    def act(self, obs):
        logits = self.pi(torch.tensor(obs, dtype=torch.float32))
        return torch.argmax(logits).item()

    def learn(self, states, actions):
        policy = Categorical(logits=self.pi(states))

        # simply maximize log likelihood of selection the expert's actions
        log_prob = policy.log_prob(actions)
        loss = - log_prob.sum()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item()


class GAIL:
    def __init__(self, dim_s, dim_a, lr, K, clip_eps, entropy_weight):
        self.dim_a = dim_a
        self.K = K
        self.clip_eps = clip_eps
        self.entropy_weight = entropy_weight
        self.discriminator = nn.Sequential(
            nn.Linear(dim_s + dim_a, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.pi = nn.Sequential(
            nn.Linear(dim_s, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, dim_a)
        )
        self.v = nn.Sequential(
            nn.Linear(dim_s, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        self.optimizer_pi = torch.optim.Adam(self.pi.parameters(), lr=lr)
        self.optimizer_v = torch.optim.Adam(self.v.parameters(), lr=lr)

    def act(self, obs):
        logits = self.pi(torch.from_numpy(obs).float())
        multinomial = Categorical(logits=logits)
        return multinomial.sample().item()

    def learn(self, batch_agent, batch_expert):
        obs, act, r_cumulated, obs_next, _ = batch_agent
        act_one_hot = F.one_hot(act.flatten(), num_classes=self.dim_a)
        obs_expert, act_expert = batch_expert
        act_expert_one_hot = F.one_hot(act_expert, num_classes=self.dim_a)

        # --- Discriminator step
        input_expert = torch.cat([obs_expert, act_expert_one_hot], dim=1)
        input_agent = torch.cat([obs, act_one_hot], dim=1)
        noise_expert = torch.normal(0, 0.01, size=input_expert.shape)
        noise_agent = torch.normal(0, 0.01, size=input_agent.shape)
        d_expert = torch.sigmoid(self.discriminator(input_expert + noise_expert))
        d_agent = torch.sigmoid(self.discriminator(input_agent + noise_agent))
        loss_d = (F.binary_cross_entropy_with_logits(d_expert, torch.ones_like(d_expert)) +
                  F.binary_cross_entropy_with_logits(d_agent, torch.zeros_like(d_agent)))
        assert not loss_d.isnan()
        self.optimizer_d.zero_grad()
        loss_d.backward()
        self.optimizer_d.step()

        # --- Policy step (with KL clipped PPO)
        # probas of the policy used to obtain the samples
        pi_k_logits = self.pi(obs).detach()
        pi_k = F.softmax(pi_k_logits, dim=1)
        pi_k_a = pi_k.gather(1, act)
        loss_pi_list = []
        loss_v_list = []
        entropy_list = []

        # rewards_clipped = torch.clamp(torch.log(d_agent), min=-100, max=0)
        # td_target = (rewards_clipped + 0.99 * self.v(obs_next) * not_finished).detach()
        # advantage = (td_target - self.v(obs)).detach()
        advantage = (r_cumulated - self.v(obs)).detach()
        for s in range(self.K):
            # train pi
            pi_logits = self.pi(obs)
            pi = F.softmax(pi_logits, dim=1)
            pi_a = pi.gather(1, act)

            ratio = pi_a / pi_k_a
            loss_1 = (ratio * advantage)
            loss_2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantage
            entropy = Categorical(pi).entropy()  # entropy term that we wish to maximize to push for exploration
            loss_pi = (-torch.min(loss_1, loss_2) - self.entropy_weight * entropy).mean()
            assert not loss_pi.isnan()
            self.optimizer_pi.zero_grad()
            loss_pi.backward()
            self.optimizer_pi.step()

            # train v_pi
            loss_v = F.smooth_l1_loss(self.v(obs), r_cumulated)
            self.optimizer_v.zero_grad()
            loss_v.backward()
            self.optimizer_v.step()

            loss_pi_list.append(loss_pi.item())
            loss_v_list.append(loss_v.item())
            entropy_list.append(entropy.mean().item())

        return {'loss_discriminator': loss_d.item(),
                'loss_actor': np.mean(loss_pi_list),
                'loss_critic': np.mean(loss_v_list),
                'd_expert': d_expert.mean().item(),
                'd_agent': d_agent.mean().item(),
                'entropy': np.mean(entropy_list)}


def run_test(agent, episodes: int, env: gym.Env):
    rewards = []
    for episode in range(episodes):
        obs = env.reset()
        done = False
        r_sum = 0
        while not done:
            # select action
            action = agent.act(obs)
            # act, observe reward and the transition
            next_obs, reward, done, info = env.step(action)
            obs = next_obs.copy()
            r_sum += reward
        rewards.append(r_sum)
    return rewards


def train_behavioral_cloning(conf, agent: BehavioralCloning, train_dataloader: DataLoader, env: gym.Env,
                             writer: SummaryWriter):
    global_step = 0
    for epoch in tqdm(range(conf.epochs)):
        for states, actions in train_dataloader:
            loss = agent.learn(states=states, actions=actions)
            global_step += 1

            if global_step % conf.log_every_n_steps == 0:
                writer.add_scalar('train_loss', loss, global_step)

            if global_step % conf.test_freq == 0:
                rewards = run_test(agent, conf.test_episodes, env)
                writer.add_scalar('reward', np.mean(rewards), epoch)


def train_gail(conf, agent: GAIL, env: gym.Env, train_dataloader: DataLoader, writer: SummaryWriter):
    global_step = 0
    for epoch in tqdm(range(conf.epochs)):
        writer.add_scalar('epoch', epoch, global_step)
        memory = Memory(conf.batch_size)
        # collect batch_size transitions from trajectories of the current policy
        while len(memory) < conf.train_every_n_events:
            obs = env.reset()
            done = False
            while not done:
                # act and observe the next step
                act = agent.act(obs)
                obs_next, reward, done, info = env.step(act)
                memory.store(transition=(obs, act, reward, obs_next, done, info))
                obs = obs_next.copy()

        for i in range(conf.train_iter):
            memory.compute_cumulated_r(agent, agent.dim_a)
            loss_dict = agent.learn(batch_agent=memory.get_minibatch_proxy_reward(),
                                    batch_expert=next(iter(train_dataloader)))
            global_step += 1
            if global_step % conf.log_every_n_steps == 0:
                for key, value in loss_dict.items():
                    writer.add_scalar(key, value, global_step)

        if global_step % conf.test_freq == 0:
            rewards = run_test(agent, conf.test_episodes, env)
            writer.add_scalar('reward', np.mean(rewards), global_step)


def main():
    project_dir = Path(__file__).resolve().parents[0]
    conf = OmegaConf.load(project_dir.joinpath('conf.yaml'))
    time_tag = datetime.datetime.now().strftime(f'%Y%m%d-%H%M%S')
    log_dir = f'runs/tp12-{conf.algo}-{time_tag}-seed{conf.seed}'
    writer = SummaryWriter(log_dir)
    save_src_and_config(log_dir, conf, writer)

    # seed the env
    seed_everything(conf.seed)
    env = gym.make('LunarLander-v2')
    env.seed(conf.seed)
    env.action_space.seed(conf.seed)
    print(f'{log_dir}')
    print(f'Training agent with conf :\n{OmegaConf.to_yaml(conf)}')

    # dimensions of the env
    dim_s = 8
    dim_a = 4

    expert = ExpertAgent(dim_a, dim_s, project_dir / 'expert.pkl')

    train_dataset = TensorDataset(expert.states, expert.actions)
    train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size)

    if conf.algo == 'bc':
        agent = BehavioralCloning(dim_s=dim_s, dim_a=dim_a, lr=conf.lr)
        train_behavioral_cloning(conf, agent, train_dataloader, env, writer)
    elif conf.algo == 'gail':
        agent = GAIL(dim_s=dim_s, dim_a=dim_a, lr=conf.lr, K=conf.K, clip_eps=conf.clip_eps,
                     entropy_weight=conf.entropy_weight)
        train_gail(conf, agent, env, train_dataloader, writer)


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
