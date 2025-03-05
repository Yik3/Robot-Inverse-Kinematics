# FCN Net
'''
A Fully Connect NN. 
Now use dropout and batchnorm. May write another one without this.
A purely FCN for 2D joint angle prediction
Written by Yike Shi on 03/01/2025
'''
import torch
import torch.nn as nn


class FCN(nn.Module):
    def __init__(self, input_size=7, hidden_sizes=[1024, 2048, 1024, 512], output_size=3, dropout_rate=0.2):
        super(FCN, self).__init__()
        
        # Fully connected layers with batch normalization and dropout
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])  
        self.dropout1 = nn.Dropout(dropout_rate)    

        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        self.dropout3 = nn.Dropout(dropout_rate)

        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.bn4 = nn.BatchNorm1d(hidden_sizes[3])
        self.dropout4 = nn.Dropout(dropout_rate)
        
        self.fc8 = nn.Linear(hidden_sizes[3], output_size)  # Output layer
        # Activation function
        self.relu = nn.SiLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        x = self.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)

        x = self.fc8(x)  # Output layer (No activation for regression task)
        x = self.tanh(x)
        return x
    

class ResNetFCN(nn.Module):
    def __init__(self, input_size=7, hidden_sizes=[1024, 1024, 512, 512], output_size=3, dropout_rate=0.1):
        super(ResNetFCN, self).__init__()
        
        # First Layer (No Skip Connection)
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.dropout1 = nn.Dropout(dropout_rate)

        # Residual Block 1
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.skip1 = nn.Linear(hidden_sizes[0], hidden_sizes[1])  # Projection layer for residual connection

        # Residual Block 2
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        self.skip2 = nn.Linear(hidden_sizes[1], hidden_sizes[2])  # Projection layer

        # Residual Block 3
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.bn4 = nn.BatchNorm1d(hidden_sizes[3])
        self.skip3 = nn.Linear(hidden_sizes[2], hidden_sizes[3])  # Projection layer

        # Output Layer
        self.fc_out = nn.Linear(hidden_sizes[3], output_size)

        # Activation Functions
        self.activation = nn.SiLU()  # Swish activation
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Initial Layer
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        # Residual Block 1
        residual = self.skip1(x)  # Align residual dimension
        x = self.activation(self.bn2(self.fc2(x)))
        x += residual  # Add skip connection

        # Residual Block 2
        residual = self.skip2(x)  # Align residual dimension
        x = self.activation(self.bn3(self.fc3(x)))
        x += residual  # Add skip connection

        # Residual Block 3
        residual = self.skip3(x)  # Align residual dimension
        x = self.activation(self.bn4(self.fc4(x)))
        x += residual  # Add skip connection

        # Output Layer
        x = self.fc_out(x)
        x = self.tanh(x)

        return x
