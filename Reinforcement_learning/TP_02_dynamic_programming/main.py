import matplotlib
from omegaconf import OmegaConf
import gym
import gridworld
import numpy as np
import pytorch_lightning as pl
import pandas as pd

SEED = 42
pl.seed_everything(SEED)
matplotlib.use("TkAgg")


def compute_greedy_pi(V, states, actions, P, discount):
    """
    Compute the greedy policy relative to V. If V is V*, then pi will be the optimal policy

    Args:
        V:
        states:
        actions:
        P:
        discount:

    Returns:
        pi: array of len N_states, containing the action to execute

    """
    reward_estimates = np.zeros((len(states), len(actions)))
    for s in P.keys():
        i_s = states[s]
        for a in actions.keys():
            for transition in P[s][a]:
                p, s_prime, r, done = transition
                i_s_prime = states[s_prime]
                reward_estimates[i_s, a] += p * (r + discount * V[i_s_prime])
    pi = np.argmax(reward_estimates, axis=1)

    return pi


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()


class ValueIterationAgent(object):
    def __init__(self, action_space, actions, states, mdp, precision_v, discount, verbose):
        self.action_space = action_space
        self.states = states
        self.actions = actions
        self.policy, self.steps = self.get_policy(states, actions, mdp, precision_v, discount)
        if verbose:
            print(f'ValueIteration converged in {self.steps} steps on policy :\n{self.policy}')

    @staticmethod
    def get_policy(states, actions, P, epsilon, discount):
        V = np.zeros(len(states))
        converged = False
        step = 0
        # estimate the value function V of the optimal policy
        while not converged:
            step += 1
            V_new = np.zeros(len(states))
            for s in P.keys():
                i_s = states[s]
                values = np.zeros(len(actions))
                for a in actions.keys():
                    for transition in P[s][a]:
                        p, s_prime, r, done = transition
                        i_s_prime = states[s_prime]
                        values[a] += p * (r + discount * V[i_s_prime])
                V_new[i_s] = np.max(values)
            converged = np.max(np.abs(V_new - V)) < epsilon
            if step % 100 == 0:
                print(f'vi {step} - {np.max(np.abs(V_new - V))}')
            V = V_new

        # estimate the optimal policy
        pi = compute_greedy_pi(V, states, actions, P, discount)
        return pi, step

    def act(self, observation):
        i_obs = self.states[gridworld.GridworldEnv.state2str(observation)]
        return self.policy[i_obs]


class PolicyIterationAgent(object):
    """
    Obtained from the policy iteration algorithm
    """

    def __init__(self, action_space, actions, states, mdp, precision_v, discount, verbose):
        self.states = states
        self.policy, self.steps = self.get_policy(states, actions, action_space, mdp, precision_v, discount)
        if verbose:
            print(f'PolicyIteration converged in {self.steps} steps on policy :\n{self.policy}')

    @staticmethod
    def get_policy(states, actions, action_space, P, epsilon, discount):
        # start with a random policy
        pi = np.array([action_space.sample() for s in states.keys()])

        converged_pi = False
        step = 0
        while not converged_pi:
            # estimate the value function of pi
            V = np.zeros(len(states))
            converged_V = False
            while not converged_V:
                step += 1
                V_new = np.zeros(len(states))
                for s in P.keys():
                    # only consider states that are not final states (and have transitions)
                    i_s = states[s]
                    for transition in P[s][pi[i_s]]:
                        p, s_prime, r, done = transition
                        i_s_prime = states[s_prime]
                        V_new[i_s] += p * (r + discount * V[i_s_prime])
                # stopping criterion
                converged_V = np.max(np.abs(V_new - V)) < epsilon
                V = V_new
                # if step % 100 == 0:
                #     print(f'vp {step} - {np.max(np.abs(V_new - V))}')

            # update the policy
            pi_new = compute_greedy_pi(V, states, actions, P, discount)

            # stopping criterion
            converged_pi = (pi_new == pi).all()
            pi = pi_new
        return pi, step

    def act(self, observation):
        i_obs = self.states[gridworld.GridworldEnv.state2str(observation)]
        return self.policy[i_obs]


def get_results_on_map(map_nb: int, conf, run_policy_iteration=True):
    # Create the (deterministic) environment
    env = gym.make('gridworld-v0')
    env.setPlan(f'gridworldPlans/plan{map_nb}.txt', {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    env.seed(SEED)
    env.action_space.seed(SEED)  # to make sure action_space.sample() is seeded as well

    # Dynamic Programming: we know the MDP, and we will use it to train our agents offline
    #   state_dic[state_str] = state_int
    #   mdp[state_str] = transitions (from this state)
    #   transitions[action] = [(proba, new_state (str), reward, done) for new_state in possible_states]
    state_dict, mdp = env.getMDP()
    print(f'[map {map_nb}] loaded mdp')
    # here, the transitions are identical, no matter the current state : just take the first one
    _, transitions = list(mdp.items())[0]

    if conf.verbose:
        env.render(mode="human")  # visualisation sur la console

    # Initialize and train our agents, offline
    agents = {
        'random': RandomAgent(env.action_space),
        'value_iteration': ValueIterationAgent(env.action_space, env.actions, state_dict, mdp, conf.precision_v,
                                               conf.discount, conf.verbose),
    }
    if run_policy_iteration:
        agents['policy_iteration'] = PolicyIterationAgent(env.action_space, env.actions, state_dict, mdp,
                                                          conf.precision_v, conf.discount, conf.verbose)

    r_sums = {}
    ep_lengths = {}
    # Test the agents
    print(f'[map {map_nb}] testing agents on {conf.episode_count} episodes..')
    for agent_name, agent in agents.items():
        r_sums[agent_name] = []
        ep_lengths[agent_name] = []
        for ep in range(conf.episode_count):
            t = 0
            r_sum = 0
            obs = env.reset()
            env.verbose = conf.verbose and ep % 200 == 0
            # if env.verbose:
            #     env.render(conf.FPS)

            # Run an episode
            while True:
                action = agent.act(obs)
                obs, reward, done, _ = env.step(action)
                r_sum += reward
                t += 1
                # if env.verbose:
                #     env.render(conf.FPS)
                if done:
                    r_sums[agent_name].append(r_sum)
                    ep_lengths[agent_name].append(t)
                    break
    env.close()

    if conf.verbose:
        for agent_name in agents.keys():
            print(f'{agent_name} - avg reward : {np.mean(r_sums[agent_name]):.4f} ± {np.std(r_sums[agent_name]):.4f}')
            print(f'{agent_name} - avg nb of actions : {np.mean(ep_lengths[agent_name]):.1f} ± '
                  f'{np.std(ep_lengths[agent_name]):.1f}')
    if run_policy_iteration:
        return r_sums, ep_lengths, len(state_dict), agents['value_iteration'].steps, agents['policy_iteration'].steps
    else:
        return r_sums, ep_lengths, len(state_dict), agents['value_iteration'].steps


def main():
    conf = OmegaConf.load('config.yaml')

    # Simply run the algos on map 0
    # get_results_on_map(0, conf)

    # Run the algos on all maps for comparison (except #9, for which the MDP is too complex to even compute)
    maps = list(range(11))
    maps.remove(9)
    data_frames = []
    for map_nb in maps:
        if map_nb in [4, 7]:
            # on these maps, PI does not converge
            r_sums, ep_lengths, map_state_len, steps_vi = get_results_on_map(map_nb, conf, run_policy_iteration=False)
            steps_pi = None
        else:
            r_sums, ep_lengths, map_state_len, steps_vi, steps_pi = get_results_on_map(map_nb, conf)
        df_r_sum = pd.DataFrame(r_sums).stack().reset_index().rename(
            columns={'level_0': 'ep', 'level_1': 'agent', 0: 'r_sum'})
        df_ep_len = pd.DataFrame(ep_lengths).stack().reset_index().rename(
            columns={'level_0': 'ep', 'level_1': 'agent', 0: 'ep_len'})
        df = pd.merge(df_r_sum, df_ep_len, on=['ep', 'agent'])
        df['map_nb'] = map_nb
        df['map_state_len'] = map_state_len
        df.loc[df.agent == 'policy_iteration', 'steps'] = steps_pi
        df.loc[df.agent == 'value_iteration', 'steps'] = steps_vi
        data_frames.append(df)
    df = pd.concat(data_frames)
    df.to_csv('results.csv')


if __name__ == '__main__':
    main()
