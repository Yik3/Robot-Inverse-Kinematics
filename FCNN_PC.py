import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create network
# input: [x, y, prev_x, prev_y, prev_theta1, prev_theta2, prev_theta3]
# output: [theta1, theta2, theta3]
class FCNN_PC(nn.Module):
    def __init__(self, num_inputs, dropout_rate=0.2):
        super(FCNN_PC, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 3),
            nn.Tanh()
        )
        self.link_lengths = torch.tensor([1.0, 1.0, 1.0], device=device)

    def forward(self, x):
        return self.layers(x)
    
    def forward_kinematics(self, angles):
        x = torch.sum(self.link_lengths * torch.cos(angles * torch.pi), dim=1, keepdim=True)
        y = torch.sum(self.link_lengths * torch.sin(angles * torch.pi), dim=1, keepdim=True)
        return torch.cat((x, y), dim=1)