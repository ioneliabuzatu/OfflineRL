from torch.utils.data import Dataset, DataLoader


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
