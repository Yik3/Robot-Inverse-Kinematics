import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
file_path = "3_segment_IK_dataset"  # Update with the correct file path if needed
data = pd.read_csv(file_path)

# Extract (x, y) pairs
x_vals = data["x"].values
y_vals = data["y"].values

# Create scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(x_vals, y_vals, s=5, alpha=0.6, color="blue", label="Generated (x, y) Pairs")

# Formatting
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Visualization of Generated (x, y) Pairs in 2D Plane")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
