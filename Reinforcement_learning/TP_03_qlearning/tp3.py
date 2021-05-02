import matplotlib.pyplot as plt
from tqdm import tqdm

# matplotlib.use("TkAgg")
import gym
import gridworld
from collections import defaultdict
import numpy as np
import pytorch_lightning as pl

pl.seed_everything(42069)


class RandomAgent:
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space
        self.algorithm = None

    def act(self, ep, obs):
        return self.action_space.sample()

    def learn(self, transition):
        pass


class Agent:
    def __init__(self, alpha, gamma, exploration_strategy, algo, eps_0=None, tau=None):
        self.Q = defaultdict(lambda: np.zeros(4))
        self.alpha = alpha
        self.gamma = gamma
        self.strategy = exploration_strategy
        self.algorithm = algo
        self._eps = eps_0
        self.tau = tau

    def eps(self, t):
        return self._eps / (1 + 0.01 * t)

    def act(self, ep, obs):
        obs = str(obs.tolist())
        if self.strategy == 'epsilon-greedy':
            if np.random.random() < self.eps(ep):
                return np.random.randint(0, 4)
            else:
                return np.argmax(self.Q[obs])

        elif self.strategy == 'boltzmann':
            probas = [np.exp(self.Q[obs][a] / self.tau) / np.sum(np.exp(self.Q[obs] / self.tau)) for a in range(4)]
            return np.random.choice(4, p=probas)

    def learn(self, transition, a_next=None):
        obs_prev, action, obs_next, reward, done = transition
        obs_prev = str(obs_prev.tolist())
        obs_next = str(obs_next.tolist())
        if self.algorithm == 'Q-learning':
            self.Q[obs_prev][action] += self.alpha * (
                    reward + self.gamma * np.max(self.Q[obs_next]) - self.Q[obs_prev][action])
        elif self.algorithm == 'SARSA':
            self.Q[obs_prev][action] += self.alpha * (
                    reward + self.gamma * self.Q[obs_next][a_next] - self.Q[obs_prev][action])


class DynaQAgent:
    def __init__(self, alpha, alpha_r, k, gamma, exploration_strategy, eps_0=None, tau=None):
        self.Q = defaultdict(lambda: np.zeros(4))
        self.alpha = alpha
        self.alpha_R = alpha_r
        self.k = k
        self.gamma = gamma
        self.strategy = exploration_strategy
        self.algorithm = None
        self._eps = eps_0
        self.tau = tau
        # estimates of R(s_t, a_t, s_t+1)
        self.R = defaultdict(lambda: [defaultdict(lambda: 0)] * 4)
        # and P(s_t+1 | s_t, a_t)
        self.P = defaultdict(lambda: defaultdict(lambda: np.zeros(4)))

    def eps(self, t):
        return self._eps / (1 + 0.01 * t)

    def act(self, ep, obs):
        obs = str(obs.tolist())
        if self.strategy == 'epsilon-greedy':
            if np.random.random() < self.eps(ep):
                return np.random.randint(0, 4)
            else:
                return np.argmax(self.Q[obs])

        elif self.strategy == 'boltzmann':
            probas = [np.exp(self.Q[obs][a] / self.tau) / np.sum(np.exp(self.Q[obs] / self.tau)) for a in range(4)]
            return np.random.choice(4, p=probas)

    def learn(self, transition):
        s, a, s_prime, reward, done = transition
        s = str(s.tolist())
        s_prime = str(s_prime.tolist())

        # Update the Q value function given the observed reward
        self.Q[s][a] += self.alpha * (reward + self.gamma * np.max(self.Q[s_prime]) - self.Q[s][a])

        # Update the MDP model: R(s_t, a_t, s_t+1) the reward estimate, and P(s_t+1 | s_t, a_t) the transition function
        self.R[s][a][s_prime] = reward  # set the value once and for all because the reward is deterministic in our case
        self.P[s_prime][s][a] += self.alpha_R * (1 - self.P[s_prime][s][a])
        for s_other in self.P.keys():
            self.P[s_other][s][a] += self.alpha_R * (0 - self.P[s_other][s][a])

        # Sample k state-action tuples, and update the Q value functions given the new MDP estimate
        for s_i in np.random.choice(list(self.Q.keys()), min(len(self.Q), self.k), replace=False):
            a_i = np.random.choice(4, replace=False)
            self.Q[s_i][a_i] += self.alpha * (sum([
                self.P[s_other][s_i][a_i] * (self.R[s_i][a_i][s_other] + self.gamma * np.max(self.Q[s_other]))
                for s_other in self.P.keys()
            ]) - self.Q[s_i][a_i])


def main():
    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    env.seed(42069)
    state_dict, mdp = env.getMDP()
    _, transitions = list(mdp.items())[0]

    agents = {
        'random': RandomAgent(action_space=env.action_space),
        'QAgent greedy opti': Agent(alpha=0.1, gamma=0.99, exploration_strategy='epsilon-greedy', algo='Q-learning', eps_0=0.5),
        'SARSA opti': Agent(alpha=0.8, gamma=0.99, exploration_strategy='epsilon-greedy', algo='SARSA', eps_0=0.5),
        'QAgent boltzmann opti': Agent(alpha=0.8, gamma=0.99, exploration_strategy='boltzmann', algo='Q-learning', tau=0.005),
        'SARSA boltzmann opti': Agent(alpha=0.8, gamma=0.99, exploration_strategy='boltzmann', algo='SARSA', tau=0.1),
        'DynaQAgent opti': DynaQAgent(alpha=0.8, alpha_r=0.4, k=50, gamma=0.99, exploration_strategy='epsilon-greedy', eps_0=0.5)
    }

    # Train agents, save and plot results
    plt.figure(figsize=(15, 8))
    for name, agent in agents.items():
        print(f'Training agent {name}')
        rsum_list = train_agent(agent, env, episode_count=10000)
        plt.plot([i for i in range(10000)], np.cumsum(rsum_list), label=name)
    plt.legend()
    plt.show()

    env.close()


def train_agent(agent, env, episode_count, render_env=False):
    rsum_list = []
    FPS = 0.0001
    for ep in tqdm(range(episode_count)):
        obs = env.reset()
        env.verbose = (ep % 100 == 0 and ep > 0 and render_env)  # display 1 episode out of 100
        if env.verbose:
            env.render(FPS)
        t = 0
        rsum = 0

        # Initial action
        action = agent.act(ep, obs)
        while True:
            # Take action
            obs_next, reward, done, _ = env.step(action)

            # Plan next action and learn from the previous results
            transition = (obs, action, obs_next, reward, done)
            action = agent.act(ep, obs)
            if agent.algorithm == 'SARSA':
                agent.learn(transition, a_next=action)
            else:
                agent.learn(transition)

            # Update observation
            obs = obs_next.copy()

            rsum += reward
            t += 1
            if env.verbose:
                env.render(FPS)
            if done:
                break

        rsum_list.append(rsum)
    return rsum_list


if __name__ == '__main__':
    main()
