###Process Forward Kinematics Data###
###Read in data file, transfer into degrees, and split data into Training and Testing Data

'''
Written by Yike Shi on 03/01/2025
'''
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os
import pandas as pd

def load_data():
    # Load dataset
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_inverse_kinematics_dataset.csv")
    data = np.loadtxt(file_path, skiprows=1, delimiter=',')

    # Split into input (xyz) and output (q)
    q = (data[:, :6])  # Joint angles
    xyz = data[:, 6:]  # End-effector position

    q = np.degrees(q)
    # Convert to PyTorch tensors
    joint_data = torch.from_numpy(q).float()
    end_effector_pos = torch.from_numpy(xyz).float()

    # Create dataset
    dataset = TensorDataset(end_effector_pos, joint_data)  # (xyz → q)

    # Split dataset
    train_size = int(0.87 * len(dataset))
    val_size = int(0.08 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    return train_dataset, val_dataset, test_dataset

def load_2d_data():
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3_segment_IK_dataset")
    data = pd.read_csv(file_path)

    # Extract input (xy positions) and output (joint angles)
    xy_data = data[["x", "y", "prev_x", "prev_y","prev_theta1","prev_theta2","prev_theta3"]].values  # Input (4D)
    joint_angles = data[["theta1", "theta2", "theta3"]].values  # Output (3D)

    # Convert joint angles from radians to degrees (if needed)
    joint_angles = np.degrees(joint_angles)/ 180.0

    # Convert to PyTorch tensors
    xy_tensor = torch.tensor(xy_data, dtype=torch.float32)
    joint_tensor = torch.tensor(joint_angles, dtype=torch.float32)

    # Create dataset (input: xy positions, output: joint angles)
    dataset = TensorDataset(xy_tensor, joint_tensor)

    # Split dataset into training, validation, and test sets
    train_size = int(0.85 * len(dataset))
    val_size = int(0.08 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Testing samples: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset

# Optional: Run this script standalone to test
if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = load_data()
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Testing samples: {len(test_dataset)}")
