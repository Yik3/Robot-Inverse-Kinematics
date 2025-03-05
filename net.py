# FCN Net
'''
A Fully Connect NN. 
Now use dropout and batchnorm. May write another one without this.

Written by Yike Shi on 03/01/2025
'''
import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, input_size=3, hidden_sizes=[1024, 2048, 4096, 8192, 16384, 8192,2048], output_size=6, dropout_rate=0.2):
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

        self.fc5 = nn.Linear(hidden_sizes[3], hidden_sizes[4])
        self.bn5 = nn.BatchNorm1d(hidden_sizes[4])
        self.dropout5 = nn.Dropout(dropout_rate)

        self.fc6 = nn.Linear(hidden_sizes[4], hidden_sizes[5])
        self.bn6 = nn.BatchNorm1d(hidden_sizes[5])
        self.dropout6 = nn.Dropout(dropout_rate)

        self.fc7 = nn.Linear(hidden_sizes[5], hidden_sizes[6])  # Output layer
        self.bn7 = nn.BatchNorm1d(hidden_sizes[6])
        self.dropout7 = nn.Dropout(dropout_rate)
        
        self.fc8 = nn.Linear(hidden_sizes[6], output_size)  # Output layer
        # Activation function
        self.relu = nn.PReLU()
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

        x = self.relu(self.bn5(self.fc5(x)))
        x = self.dropout5(x)

        x = self.relu(self.bn6(self.fc6(x)))
        x = self.dropout6(x)
        
        x = self.relu(self.bn7(self.fc7(x)))
        x = self.dropout7(x)

        x = self.fc8(x)  # Output layer (No activation for regression task)
        x = self.tanh(x) * 180.0
        return x
    

class LSTMNetwork(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, output_size=6):
        super(LSTMNetwork, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)  # (batch, seq_len=1, input_size)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, output_size)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # LSTM forward pass
        x, _ = self.lstm(x)  # Output shape: (batch, seq_len, hidden_size)

        # Extract the last timestep output (since seq_len = 1, we take lstm_out[:, -1, :])
        #x = lstm_out[:, -1, :]

        # Pass through fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # Final output (joint angles)

        return x

class FCN_FWD(nn.Module):
    def __init__(self, input_size=6, hidden_sizes=[128,256,64], output_size=3, dropout_rate=0.2):
        super(FCN_FWD, self).__init__()
        
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
        
        self.fc8 = nn.Linear(hidden_sizes[2], output_size)  # Output layer
        # Activation function
        self.relu = nn.ReLU()
        

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        x = self.fc8(x)  # Output layer (No activation for regression task)
        
        return x