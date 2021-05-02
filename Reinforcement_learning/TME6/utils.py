import glob
import shutil

import numpy as np
from omegaconf import OmegaConf


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


class FeatureExtractor(object):
    def __init__(self):
        super().__init__()

    def get_features(self, obs):
        pass


class NothingToDo(FeatureExtractor):
    def __init__(self, env):
        super().__init__()
        obs = env.reset()
        self.out_size = len(obs)

    def get_features(self, obs):
        return obs


class MapFromDumpExtractor(FeatureExtractor):
    def __init__(self, env):
        super().__init__()
        out_size = env.start_grid_map.reshape(1, -1).shape[1]
        self.out_size = out_size

    def get_features(self, obs):
        # prs(obs)
        return obs.reshape(-1)


class MapFromDumpExtractor2(FeatureExtractor):
    def __init__(self, env):
        super().__init__()
        out_size = env.start_grid_map.reshape(1, -1).shape[1]
        self.out_size = out_size * 3

    def get_features(self, obs):
        state = np.zeros((3, np.shape(obs)[0], np.shape(obs)[1]))
        state[0] = np.where(obs == 2, 1, state[0])
        state[1] = np.where(obs == 4, 1, state[1])
        state[2] = np.where(obs == 6, 1, state[2])
        return state.reshape(-1)


class DistsFromStates(FeatureExtractor):
    def __init__(self, env):
        super().__init__()
        self.out_size = 16

    def get_features(self, obs):
        # prs(obs)
        # x=np.loads(obs)
        x = obs
        # print(x)
        astate = list(map(
            lambda x: x[0] if len(x) > 0 else None,
            np.where(x == 2)
        ))
        astate = np.array(astate)
        a3 = np.where(x == 3)
        d3 = np.array([0])
        if len(a3[0]) > 0:
            astate3 = np.concatenate(a3).reshape(2, -1).T
            d3 = np.power(astate - astate3, 2).sum(1).min().reshape(1)

            # d3 = np.array(d3).reshape(1)
        a4 = np.where(x == 4)
        d4 = np.array([0])
        if len(a4[0]) > 0:
            astate4 = np.concatenate(a4).reshape(2, -1).T
            d4 = np.power(astate - astate4, 2).sum(1).min().reshape(1)
            # d4 = np.array(d4)
        a5 = np.where(x == 5)
        d5 = np.array([0])
        # prs(a5)
        if len(a5[0]) > 0:
            astate5 = np.concatenate(a5).reshape(2, -1).T
            d5 = np.power(astate - astate5, 2).sum(1).min().reshape(1)
            # d5 = np.array(d5)
        a6 = np.where(x == 6)
        d6 = np.array([0])
        if len(a6[0]) > 0:
            astate6 = np.concatenate(a6).reshape(2, -1).T
            d6 = np.power(astate - astate6, 2).sum(1).min().reshape(1)
            # d6=np.array(d6)

        # prs("::",d3,d4,d5,d6)
        ret = np.concatenate((d3, d4, d5, d6)).reshape(1, -1)
        ret = np.dot(ret.T, ret)
        return ret.reshape(-1)
