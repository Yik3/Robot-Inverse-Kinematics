from FCNN_PC import FCNN_PC
from OnlyPositionDataset import OnlyPositionDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# HYPERPARAMETERS
num_epochs = 100
lr = 0.0005
batch_size = 256
dropout_rate = 0.3
alpha = 1  # how heavily to weigh position loss (vs angle loss)

# create dataset and dataloaders
dataset = OnlyPositionDataset("3_segment_IK_dataset")
train_size = int(0.64 * len(dataset))
val_size = int(0.16 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

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

model = FCNN_PC(2, dropout_rate).to(device)
mae_loss = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

training_losses = []
validation_losses = []

print("Training...")
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    model.train()
    total_train_loss = 0
    for input_position, target_angles in train_loader:
        input_position, target_angles = input_position.to(device), target_angles.to(device)
        output_angles = model(input_position)
        output_position = model.forward_kinematics(output_angles)
        loss = alpha * mae_loss(output_position, input_position) + (1 - alpha) * mae_loss(output_angles, target_angles)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)
    training_losses.append(avg_train_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss}")

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for input_position, target_angles in val_loader:
            input_position, target_angles = input_position.to(device), target_angles.to(device)
            output_angles = model(input_position)
            output_position = model.forward_kinematics(output_angles)
            loss = alpha * mae_loss(output_position, input_position) + (1 - alpha) * mae_loss(output_angles, target_angles)
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader)
    validation_losses.append(avg_val_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss}")


torch.save(model.state_dict(), "best_only_position_FCNN_PC_model.pth")
model.eval()
test_loss = 0

with torch.no_grad():
    for input_position, target_angles in test_loader:
        input_position, target_angles = input_position.to(device), target_angles.to(device)
        output_angles = model(input_position)
        output_position = model.forward_kinematics(output_angles)
        loss = alpha * mae_loss(output_position, input_position) + (1 - alpha) * mae_loss(output_angles, target_angles)
        test_loss += loss.item()

avg_test_loss = test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss}")
        
plt.figure()
plt.plot(training_losses, label="Training Loss")
plt.plot(validation_losses, label="Validation Loss")
plt.show()

torch.save(model.state_dict(), "random_harry_testing_stuffs/best_only_position_FCNN_PC_model.pth")
