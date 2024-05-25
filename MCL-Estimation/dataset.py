import os

import numpy as np
import torch
from torch.utils.data import Dataset


class MCLDataset(Dataset):

    def __init__(self, root_dir, train):
        super(MCLDataset, self).__init__()
        if train:
            self.mcl = np.load(os.path.join(root_dir, 'mcl_train.npy'))
            self.angle = np.load(os.path.join(root_dir, 'angle_train.npy'))
            self.acc = np.load(os.path.join(root_dir, 'acc_train.npy'))
        else:
            self.mcl = np.load(os.path.join(root_dir, 'mcl_test.npy'))
            self.angle = np.load(os.path.join(root_dir, 'angle_test.npy'))
            self.acc = np.load(os.path.join(root_dir, 'acc_test.npy'))

    def __len__(self):
        return len(self.mcl)

    def __getitem__(self, idx):
        mcl = self.mcl[idx]
        angle = self.angle[idx]
        acc = self.acc[idx]
        return torch.from_numpy(mcl), torch.from_numpy(angle), torch.from_numpy(acc)
