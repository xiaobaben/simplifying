# Time : 2023/2/16 13:20
# Author : 小霸奔
# FileName: dataloader.p
# Time : 2022/11/28 19:10
# Author : 小霸奔
# FileName: eegeog_ISRUC_dataloader.p
from torch.utils.data import Dataset
import numpy as np
import torch

"""
[ "E1-M2", "E2-M2", "F4-M1", "F3-M2", "C4-M1", "C3-M2", "O1-M2", "O2-M1"]
"""


class Build_loader(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.train_len = len(self.data_path[0])

    def __getitem__(self, index):
        x_data = np.load(self.data_path[0][index])
        y_data = np.load(self.data_path[1][index])
        x_data = torch.from_numpy(np.array(x_data).astype(np.float32))
        y_data = torch.from_numpy(np.array(y_data).astype(np.float32))
        eog = x_data[:, :2, :]
        eeg = x_data[:, 2:, :]
        mean = torch.mean(eog)
        std = torch.std(eog)
        gaussian_noise = torch.from_numpy(np.random.normal(loc=mean, scale=std, size=(20, 100)).astype(np.float32))

        return eog, eeg, gaussian_noise, y_data

    def __len__(self):
        return self.train_len




#


