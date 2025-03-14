import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from process_data import *
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3_segment_IK_dataset")
data = pd.read_csv(file_path)
xy_data = data[["x", "y", "prev_x", "prev_y","prev_theta1","prev_theta2","prev_theta3"]].values  # Input (4D)
joint_angles = data[["theta1", "theta2", "theta3"]].values
joint_angles = np.degrees(joint_angles)/ 180.0

X_train, X_test, y_train, y_test = train_test_split(xy_data, joint_angles, test_size=0.1, random_state=42)

degrees = range(1, 7)
test_errors = []

# Iterate over polynomial degrees
for degree in degrees:
    # Transform features into polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Fit Linear Regression model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Predict on test set
    y_pred = model.predict(X_test_poly)

    # Compute Mean Squared Error (MSE) on test set
    mse = mean_absolute_error(y_test, y_pred)
    test_errors.append(mse)
    print(f"Degree {degree}: Test MAE = {mse:.6f}")

# Plot the test errors
plt.figure(figsize=(8, 5))
plt.plot(degrees, test_errors, marker='o', linestyle='-')
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Absolute Error (MAE)")
plt.title("Testing Error vs Polynomial Degree")
plt.grid(True)
plt.show()