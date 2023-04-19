import torch
from torch import nn
from torch.utils.data import Dataset

class ONNDataSet(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]
        # label = [l[idx] for l in self.labels]
        label = torch.cat(self.labels, dim=1)[idx]

        return sample, label