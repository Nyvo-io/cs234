import random
import numpy as np
import torch


class Dataset:
    def __init__(self, *components):
        self.n = len(components[0])
        for c in components:
            assert len(c) == self.n

        # components: [obs1, obs2 (same as obs1), act1, act2, label]
        self.components = [torch.from_numpy(c).to(torch.float32) for c in components]
        act1 = self.components[2]
        act2 = self.components[3]
        unlabel_act = torch.zeros_like(act1)
        unlabel_act[0::2] = act1[0::2]
        unlabel_act[1::2] = act2[1::2]
        self.components.append(unlabel_act)

    def sample(self, batch_size):
        indices = np.random.choice(self.n, size=batch_size, replace=False)
        return tuple(c[indices] for c in self.components)


def load_data(path, strict_pref_only=False):
    npz = np.load(path, allow_pickle=True)
    if not strict_pref_only:
        return Dataset(npz["obs_1"], npz["obs_2"], npz["action_1"], npz["action_2"], npz["label"])
    idxs = npz["label"] != 0.5
    return Dataset(
        npz["obs_1"][idxs],
        npz["obs_2"][idxs],
        npz["action_1"][idxs],
        npz["action_2"][idxs],
        npz["label"][idxs],
    )


def get_trajectories(env):
    dataset = env.get_dataset()
    dones = dataset["terminals"] | dataset["timeouts"]
    done_indices = dones.nonzero()[0] + 1

    keys = ["observations", "actions", "rewards", "infos/qpos", "infos/qvel"]
    zipped = list(zip(*[np.split(dataset[k], done_indices)[:-1] for k in keys]))
    return [dict(zip(keys, vals)) for vals in zipped]


def sample_segment(trajectories, segment_len):
    traj = random.choice(trajectories)
    T = len(traj["observations"])
    start_t = random.randrange(T - segment_len)
    end_t = start_t + segment_len
    return {k: v[start_t:end_t] for k, v in traj.items()}
