import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        # convert to torch tensor
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)  # Use long for classification

        # normulization (min max scaling)
        X_min = self.X.min()
        X_max = self.X.max()
        self.X = (self.X - X_min) / (X_max - X_min)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]