import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

df = pd.read_csv("3_segment_IK_dataset")
data = df.to_numpy()



class ThreeSegmentDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.data = df.to_numpy(dtype=np.float32)
        self.x = self.data[:, :7]
        self.y = self.data[:, 7:]

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]