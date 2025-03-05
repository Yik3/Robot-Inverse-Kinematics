import os
import pandas as pd
import numpy as np

def check_non_mapping_data(angle_threshold=5.0,name="3_segment_IK_dataset"):
    """
    Checks for non-mapping issues in the dataset:
    1. If the same (x, y) results in different joint angles.
    2. If the same (x, y, prev_x, prev_y) results in different joint angles.
    
    Args:
        angle_threshold (float): Maximum allowed deviation (in degrees) between solutions to be considered identical.
    """
    # Load dataset
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    data = pd.read_csv(file_path)

    # Convert joint angles from radians to degrees (if needed)
    data[["theta1", "theta2", "theta3"]] = np.degrees(data[["theta1", "theta2", "theta3"]])

    ### **Scenario 1: Checking (x, y) → Joint Angle Variability**
    bad_xy_cases = []
    
    for (x, y), group in data.groupby(["x", "y"]):
        joint_angles = group[["theta1", "theta2", "theta3"]].values
        max_diff = np.max(np.abs(joint_angles - joint_angles.mean(axis=0)), axis=0)  # Max deviation from mean
        
        if np.any(max_diff > angle_threshold):  # If any angle exceeds threshold
            bad_xy_cases.append((x, y, max_diff))
    
    ### **Scenario 2: Checking (x, y, prev_x, prev_y) → Joint Angle Variability**
    bad_xy_prev_cases = []

    for (x, y, prev_x, prev_y), group in data.groupby(["x", "y", "prev_x", "prev_y"]):
        joint_angles = group[["theta1", "theta2", "theta3"]].values
        max_diff = np.max(np.abs(joint_angles - joint_angles.mean(axis=0)), axis=0)  # Max deviation from mean

        if np.any(max_diff > angle_threshold):  # If any angle exceeds threshold
            bad_xy_prev_cases.append((x, y, prev_x, prev_y, max_diff))
    
    ### **Print Results**
    print(f"Total unique (x, y) pairs: {data.groupby(['x', 'y']).ngroups}")
    print(f"Cases where same (x, y) map to significantly different joint angles (> {angle_threshold}°): {len(bad_xy_cases)}")

    print(f"Total unique (x, y, prev_x, prev_y) pairs: {data.groupby(['x', 'y', 'prev_x', 'prev_y']).ngroups}")
    print(f"Cases where same (x, y, prev_x, prev_y) map to significantly different joint angles (> {angle_threshold}°): {len(bad_xy_prev_cases)}")

    if bad_xy_cases:
        print("\nExample cases where same (x, y) have significantly different joint angles:")
        for case in bad_xy_cases[:5]:  # Show only a few examples
            print(f"x: {case[0]}, y: {case[1]}, Max Diff: {case[2]}")

    if bad_xy_prev_cases:
        print("\nExample cases where same (x, y, prev_x, prev_y) have significantly different joint angles:")
        for case in bad_xy_prev_cases[:5]:  # Show only a few examples
            print(f"x: {case[0]}, y: {case[1]}, prev_x: {case[2]}, prev_y: {case[3]}, Max Diff: {case[4]}")

# Run the check
check_non_mapping_data(angle_threshold=5.0)  # 5-degree threshold for significant variation
