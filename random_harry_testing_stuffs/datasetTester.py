from ThreeSegmentDataset import ThreeSegmentDataset
import torch
from torch.utils.data import DataLoader
dataset = ThreeSegmentDataset("3_segment_IK_dataset")

loader = DataLoader(dataset, batch_size=1, shuffle=True)

for inputs, target_angles, target_positions in loader:
    print(inputs)
    print(target_angles)
    print(target_positions)
    break