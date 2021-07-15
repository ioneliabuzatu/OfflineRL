import os

import numpy as np
from torch.utils.data import Dataset
from collections import namedtuple

Transition = namedtuple('Transition', ['states', 'actions', 'next_states', 'rewards', 'dones'])


class PartialDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data.states)

    def __getitem__(self, idx):
        # select stack of states
        states = self.data.states[idx]
        # select followup action, next_state, reward and done flag
        action = self.data.actions[idx]
        next_state = self.data.next_states[idx]
        reward = self.data.rewards[idx]
        done = self.data.dones[idx]

        return states, action, next_state, reward, done


class DatasetDemonstration(object):
    def __init__(self, root_path):
        assert (os.path.exists(root_path))
        self.root_path = root_path
        # assign list of data files found in the data root directory
        self.data_files = sorted(os.listdir(root_path))

    def __len__(self):
        # this count returns the number of files in the data root folder
        return len(self.data_files)

    def load(self, idx):
        # select an index at random from all files
        file_name = self.data_files[idx]
        file_path = os.path.join(self.root_path, file_name)
        # load the selected file
        data = np.load(file_path)
        # get the respective properties from the files
        states = data["states"]
        actions = data["actions"]
        next_states = data["next_states"]
        rewards = data["rewards"]
        dones = data["dones"]
        # clean the memory from the data file
        del data
        # return the transitions
        return Transition(states=states, actions=actions, next_states=next_states, rewards=rewards, dones=dones)

    def load_multiple_files(self, indices: list):
        multiple_states = np.zeros(shape=(2000 * len(indices), 4, 96, 96))
        multiple_actions = np.zeros(2000 * len(indices))
        multiple_next_states = np.zeros(shape=(2000 * len(indices), 4, 96, 96))
        multiple_rewards = np.zeros(2000 * len(indices))
        multiple_dones = np.zeros(2000 * len(indices))

        for i, idx in enumerate(indices):
            file_name = self.data_files[idx]
            file_path = os.path.join(self.root_path, file_name)
            data = np.load(file_path)
            multiple_states[i * 2000:i * 2000 + 2000] = data["states"]
            multiple_actions[i * 2000:i * 2000 + 2000] = data["actions"]
            multiple_next_states[i * 2000:i * 2000 + 2000] = data["next_states"]
            multiple_rewards[i * 2000:i * 2000 + 2000] = data["rewards"]
            multiple_dones[i * 2000:i * 2000 + 2000] = data["dones"]
            del data

        return Transition(states=multiple_states, actions=multiple_actions, next_states=multiple_next_states,
                          rewards=multiple_rewards, dones=multiple_dones)
