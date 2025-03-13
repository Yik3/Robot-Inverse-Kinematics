
from ThreeSegmentDataset import ThreeSegmentDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# HYPERPARAMETERS
num_epochs = 100
lr = 0.0005
batch_size = 256
dropout_rate = 0.3
alpha = 0.9 # how heavily to weigh position loss (vs angle loss)


# create dataset and dataloaders

dataset = ThreeSegmentDataset("3_segment_IK_dataset")
train_size = int(0.64 * len(dataset))
val_size = int(0.16 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True,
)



# create network
# input: [x, y, prev_x, prev_y, prev_theta1, prev_theta2, prev_theta3]
# output: [theta1, theta2, theta3]
class FCNN_PC(nn.Module):
    def __init__(self, dropout_rate):
        super(FCNN_PC, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(7, 1024),
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
    


model = FCNN_PC(dropout_rate).to(device)
mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

training_losses = []
validation_losses = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}!")
    model.train()
    train_loss = 0.0

    for inputs, target_angles, target_positions in train_loader:
        inputs = inputs.to(device)
        target_angles = target_angles.to(device)
        target_positions = target_positions.to(device)

        predicted_angles = model(inputs)
        predicted_positions = model.forward_kinematics(predicted_angles)
        
        angle_loss = mae_loss(predicted_angles, target_angles) * (1 - alpha)
        position_loss = mae_loss(predicted_positions, target_positions) * alpha
        total_loss = angle_loss + position_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        train_loss += total_loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f}")
    training_losses.append(avg_train_loss)


    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for inputs, target_angles, target_positions in val_loader:
            inputs = inputs.to(device)
            target_angles = target_angles.to(device)
            target_positions = target_positions.to(device)

            predicted_angles = model(inputs)
            predicted_positions = model.forward_kinematics(predicted_angles)
            
            angle_loss = mae_loss(predicted_angles, target_angles) * (1 - alpha)
            position_loss = mae_loss(predicted_positions, target_positions) * alpha
            total_loss = angle_loss + position_loss
            val_loss += total_loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{num_epochs} | Val Loss: {avg_val_loss:.4f}")
    validation_losses.append(avg_val_loss)

    

model.eval()
test_mse_loss = 0.0
test_mae_loss = 0.0

with torch.no_grad():
    for inputs, target_angles, target_positions in test_loader:
        inputs = inputs.to(device)
        target_angles = target_angles.to(device)

        predicted_angles = model(inputs)
        test_mse_loss += mse_loss(predicted_angles, target_angles)
        test_mae_loss += mae_loss(predicted_angles, target_angles)

avg_test_mse_loss = test_mse_loss / len(test_loader)
avg_test_mae_loss = test_mae_loss / len(test_loader)
print(f"Test MSE Loss: {avg_test_mse_loss:.4f}")
print(f"Test MAE Loss: {avg_test_mae_loss:.4f}")

plt.figure()
plt.plot(training_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.show()

        

