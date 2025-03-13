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
        self.angles = self.data[:, 4:]
        self.angles /= np.pi
        self.inputs = self.data[:, :7]
        self.target_angles = self.data[:, 7:]
        self.target_positions = self.data[:, :2]

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.target_angles[idx], self.target_positions[idx]