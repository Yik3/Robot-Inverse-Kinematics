'''
Predicting Angles of different joints of a Robot Arm
'''

from process_data import load_data
from net import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

train_dataset, val_dataset, test_dataset = load_data()
if torch.cuda.is_available():
    device=torch.device("cuda")
    print("CUDA!")
else:
    device=torch.device("cpu")
#Use Data Loader to load a batch of Data
batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = FCN(input_size=3, output_size=6, dropout_rate=0.2).to(device)  # Move model to GPU if available
model_fwd = FCN_FWD(input_size=6, output_size=3, dropout_rate=0.2).to(device)
model_fwd.load_state_dict(torch.load("best_model_fwd.pth"))  # Load trained weights
model_fwd.eval() 
criterion = nn.L1Loss()  # Mean Squared Error (MSE) for regression
criterion2 = nn.MSELoss()
L2_factor = 0.8
penalty_factor = 1500
optimizer = optim.Adam(model.parameters(), lr=0.0007, weight_decay=5e-5)  # L2 Regularization

# Training settings
num_epochs = 20  # Number of training epochs #need to increase
best_val_loss = float('inf')  # Track best validation loss
early_stop_patience = 15  # Stop training if no improvement in X epochs
early_stop_counter = 0  

# Store training/validation loss for visualization
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    ### === Training Phase === ###
    model.train()  # Set model to training mode (Dropout ON)
    train_loss = 0.0
    for xyz_batch, joint_batch in train_loader:
        xyz_batch, joint_batch = xyz_batch.to(device), joint_batch.to(device)  # Move to GPU if available

        optimizer.zero_grad()  # Clear previous gradients
        output = model(xyz_batch)  # Forward pass
        predict_xyz = model_fwd(output)
        loss = penalty_factor*(0.5*criterion2(predict_xyz, xyz_batch)+0.5*criterion(predict_xyz, xyz_batch))
        loss += (1-L2_factor)*criterion(output, joint_batch) + L2_factor*criterion2(output, joint_batch)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    ### === Validation Phase === ###
    model.eval()  # Set model to evaluation mode (Dropout OFF)
    val_loss = 0.0
    with torch.no_grad():  # No need to compute gradients during validation
        for xyz_batch, joint_batch in val_loader:
            xyz_batch, joint_batch = xyz_batch.to(device), joint_batch.to(device)
            output = model(xyz_batch)
            predict_xyz = model_fwd(output)
            loss = penalty_factor*(0.5*criterion2(predict_xyz, xyz_batch)+0.5*criterion(predict_xyz, xyz_batch))
            loss += (1-L2_factor)*criterion(output, joint_batch) + L2_factor*criterion2(output, joint_batch)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    ### === Print Progress === ###
    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}")

    ### === Early Stopping === ###
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_counter = 0  # Reset counter
        torch.save(model.state_dict(), "best_model.pth")  # Save best model
    else:
        early_stop_counter += 1
        if early_stop_counter >= early_stop_patience:
            print("Early stopping triggered.")
            break  # Stop training

print("Training complete! Best validation loss:", best_val_loss)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss", marker='o')
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker='s')
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training & Validation Loss vs. Epochs")
plt.legend()
plt.grid()
plt.show()

model.load_state_dict(torch.load("best_model.pth"))
model.eval()  # Set model to evaluation mode (Dropout OFF)

test_loss = 0.0
mae_loss = nn.L1Loss()  # Mean Absolute Error (MAE) as accuracy metric
total_mae = 0.0

with torch.no_grad():
    for xyz_batch, joint_batch in test_loader:
        xyz_batch, joint_batch = xyz_batch.to(device), joint_batch.to(device)
        output = model(xyz_batch)
        
        # Compute test loss (MSE)
        loss = loss = (1-L2_factor)*criterion(output, joint_batch) + L2_factor*criterion2(output, joint_batch)
        test_loss += loss.item()

        # Compute Mean Absolute Error (MAE)
        mae = mae_loss(output, joint_batch)
        total_mae += mae.item()

avg_test_loss = test_loss / len(test_loader)
avg_test_mae = total_mae / len(test_loader)

print(f"Test MSE Loss: {avg_test_loss:.6f}")
print(f"Test MAE (Mean Absolute Error - Accuracy): {avg_test_mae:.6f}")

print("Let's see an example!")
xyz_sample, q_sample = test_dataset[10]
print(xyz_sample)
xyz_sample = xyz_sample.reshape(1, -1).to(device)
predicted = model(xyz_sample)
xyz_pred = model_fwd(predicted)
print("The Reverse Prediction is ",xyz_pred)
print("The predicted Angles are:",predicted)
print("The real Angles are:",q_sample)