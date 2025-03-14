from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class OnlyPositionDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.data = df.to_numpy(dtype=np.float32)
        self.data[:, 4:] /= np.pi
        self.inputs = self.data[:, :2]
        self.targets = self.data[:, 7:]

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
    