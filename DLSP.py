'''
Classical IK Solution
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def forward_kinematics_non_cumulative(theta, lengths):
    """
    Computes the end-effector position (x, y) for a 3 DOF arm
    using non-cumulative joint angles.
    
    Args:
        theta: Array of joint angles [theta1, theta2, theta3] (radians).
        lengths: List/array of link lengths [l1, l2, l3].
    
    Returns:
        Array with the (x, y) position of the end-effector.
    """
    theta1, theta2, theta3 = theta
    l1, l2, l3 = lengths
    x = l1 * np.cos(theta1) + l2 * np.cos(theta2) + l3 * np.cos(theta3)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta2) + l3 * np.sin(theta3)
    return np.array([x, y])

def compute_jacobian_non_cumulative(theta, lengths):
    """
    Computes the 2x3 Jacobian for the non-cumulative forward kinematics.
    
    Args:
        theta: Array of joint angles [theta1, theta2, theta3] (radians).
        lengths: List/array of link lengths [l1, l2, l3].
    
    Returns:
        A 2x3 numpy array representing the Jacobian.
    """
    theta1, theta2, theta3 = theta
    l1, l2, l3 = lengths
    # Partial derivatives for x:
    dx_dtheta1 = -l1 * np.sin(theta1)
    dx_dtheta2 = -l2 * np.sin(theta2)
    dx_dtheta3 = -l3 * np.sin(theta3)
    # Partial derivatives for y:
    dy_dtheta1 = l1 * np.cos(theta1)
    dy_dtheta2 = l2 * np.cos(theta2)
    dy_dtheta3 = l3 * np.cos(theta3)
    
    J = np.array([[dx_dtheta1, dx_dtheta2, dx_dtheta3],
                  [dy_dtheta1, dy_dtheta2, dy_dtheta3]])
    return J

def damped_least_squares_inverse_kinematics_non_cumulative(target, theta_init, lengths, damping=0.1, max_iter=100, tol=1e-3):
    """
    Solves the inverse kinematics using the Damped Least Squares (DLS) method
    with non-cumulative forward kinematics.
    
    Args:
        target: Array [x_target, y_target] for desired end-effector position.
        theta_init: Initial joint angles (array of 3 values in radians).
        lengths: Link lengths [l1, l2, l3].
        damping: Damping factor for stability.
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance for the position error.
    
    Returns:
        theta: The computed joint angles.
        current_pos: End-effector position computed with the solution angles.
        error: Final position error.
    """
    theta = np.copy(theta_init)
    for i in range(max_iter):
        current_pos = forward_kinematics_non_cumulative(theta, lengths)
        error = target - current_pos
        if np.linalg.norm(error) < tol:
            break
        J = compute_jacobian_non_cumulative(theta, lengths)
        JT = J.T
        inv_term = np.linalg.inv(J @ JT + damping**2 * np.eye(2))
        dtheta = JT @ inv_term @ error
        theta = theta + dtheta
    return theta, current_pos, error

# ---  Test for a Single Data Point ---

# Sample data from your dataset:
# Target end-effector position (x, y)
target = np.array([-3.0, -3.6739403974420594e-16])
# Previous joint angles (initial guess), in radians
theta_init = np.array([-3.094242687623706, -3.048272825844311, -3.106616678139152])
# Expected (true) joint angles from the dataset:
true_thetas = np.array([-3.141592653589793, -3.141592653589793, -3.141592653589793])
# Define link lengths (example: all links of length 1.0)
lengths = [1.0, 1.0, 1.0]

# Compute the inverse kinematics solution using the non-cumulative model:
theta_solution, final_pos, final_error = damped_least_squares_inverse_kinematics_non_cumulative(
    target, theta_init, lengths, damping=0.1, max_iter=100, tol=1e-3)

file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3_segment_IK_dataset")
data = pd.read_csv(file_path)
xy_data = data[["x", "y", ]].values  # Input (4D)
theta_data = data[["prev_theta1","prev_theta2","prev_theta3"]].values
joint_angles = data[["theta1", "theta2", "theta3"]].values

# ---  Test for Multiple Data Points ---
# Iterate over each data point in the dataset
err = 0
err1 = 0
for i in range(len(xy_data)):
    target = xy_data[i]
    theta_init = theta_data[i]
    true_thetas = joint_angles[i]
    lengths = [1.0, 1.0, 1.0]
    theta_solution, final_pos, final_error = damped_least_squares_inverse_kinematics_non_cumulative(
    target, theta_init, lengths, damping=0.1, max_iter=50, tol=1e-3)
    err += np.abs(theta_solution - true_thetas)
    err1+= final_error

err1/=len(xy_data)
err /= len(xy_data)
err = np.mean(err)
print(f"Mean Joint Angle Error: {err}")
print(f"Mean Position Error: {err1}")
'''
print("Non-Cumulative Kinematics Test:")
print("Target position:", target)
print("Initial guess (radians):", theta_init)
print("Computed joint angles (radians):", theta_solution)
print("Computed end-effector position:", final_pos)
print("Final error:", final_error)
print("True joint angles (radians):", true_thetas)
'''