from traj_vis import *

data_str = """
Step 0: Theta1=0.000, Theta2=0.000, Theta3=0.000, at x=3.0, y =0.0
Step 1: Theta1=-0.003, Theta2=0.037, Theta3=0.018, at x=2.857142857142857, y =0.14285714285714285
Step 2: Theta1=-0.004, Theta2=0.047, Theta3=0.027, at x=2.7142857142857144, y =0.2857142857142857
Step 3: Theta1=0.003, Theta2=0.045, Theta3=0.030, at x=2.5714285714285716, y =0.42857142857142855
Step 4: Theta1=0.010, Theta2=0.039, Theta3=0.030, at x=2.428571428571429, y =0.5714285714285714
Step 5: Theta1=0.015, Theta2=0.035, Theta3=0.029, at x=2.2857142857142856, y =0.7142857142857142
Step 6: Theta1=0.016, Theta2=0.031, Theta3=0.029, at x=2.142857142857143, y =0.8571428571428571
Step 7: Theta1=0.016, Theta2=0.029, Theta3=0.029, at x=2.0, y =1.0
Step 8: Theta1=0.016, Theta2=0.028, Theta3=0.030, at x=1.8571428571428572, y =1.1428571428571428
Step 9: Theta1=0.018, Theta2=0.030, Theta3=0.032, at x=1.7142857142857144, y =1.2857142857142856
Step 10: Theta1=0.020, Theta2=0.033, Theta3=0.035, at x=1.5714285714285716, y =1.4285714285714284
Step 11: Theta1=0.023, Theta2=0.037, Theta3=0.040, at x=1.4285714285714286, y =1.5714285714285714
Step 12: Theta1=0.027, Theta2=0.042, Theta3=0.045, at x=1.2857142857142858, y =1.7142857142857142
Step 13: Theta1=0.031, Theta2=0.049, Theta3=0.051, at x=1.142857142857143, y =1.857142857142857
Step 14: Theta1=0.036, Theta2=0.055, Theta3=0.056, at x=1.0, y =2.0
Step 15: Theta1=0.044, Theta2=0.060, Theta3=0.062, at x=0.8571428571428572, y =2.142857142857143
Step 16: Theta1=0.054, Theta2=0.060, Theta3=0.067, at x=0.7142857142857144, y =2.2857142857142856
Step 17: Theta1=0.065, Theta2=0.053, Theta3=0.071, at x=0.5714285714285716, y =2.4285714285714284
Step 18: Theta1=0.067, Theta2=0.028, Theta3=0.065, at x=0.4285714285714288, y =2.571428571428571
Step 19: Theta1=0.033, Theta2=-0.040, Theta3=0.021, at x=0.28571428571428603, y =2.714285714285714
Step 20: Theta1=-0.044, Theta2=-0.165, Theta3=-0.082, at x=0.14285714285714324, y =2.8571428571428568
Step 21: Theta1=-0.098, Theta2=-0.297, Theta3=-0.198, at x=0.0, y =3.0
"""

def parse_angle_data(data_string):
    angle_sequence = []
    xy_sequence = []

    lines = [line.strip() for line in data_string.split('\n') if line.strip()]

    for line in lines:
        # Split line to get "Theta1=..., Theta2=..., Theta3=..., at x=..., y=..."
        # Example line: "Step 0: Theta1=0.000, Theta2=0.000, Theta3=0.000, at x=3, y=0"

        # 1) Remove the "Step X:" part
        step_info, values_part = line.split(": ", 1)

        # 2) Split the angle part from the 'at x,y' part
        angles_part, xy_part = values_part.split(", at")

        # ---- Parse the angles (convert to degrees) ----
        # angles_part like "Theta1=0.000, Theta2=0.000, Theta3=0.000"
        theta_strs = angles_part.split(", ")
        # Each t_str is e.g. "Theta1=0.000"
        angles = []
        for t_str in theta_strs:
            val = float(t_str.split("=")[1]) * 180.0  # Convert to degrees
            angles.append(val)
        angle_sequence.append(tuple(angles))  # (theta1_deg, theta2_deg, theta3_deg)

        # ---- Parse the x,y part ----
        # xy_part like " x=3, y=0"
        # remove leading " x=" or split by commas
        # e.g. after strip: "x=3, y=0"
        xy_str = xy_part.strip().split(",")
        # xy_str[0] = "x=3", xy_str[1] = " y=0"
        x_val = float(xy_str[0].split("=")[1])
        y_val = float(xy_str[1].split("=")[1])
        xy_sequence.append((x_val, y_val))

    return angle_sequence, xy_sequence


# 执行转换
theta_sequence,xy_sequence = parse_angle_data(data_str)
#print(xy_sequence)
print(theta_sequence[-1],xy_sequence[-1])

test_angles = theta_sequence
    
    # 初始化可视化工具（使用绝对角度模式）
vis = ArmVisualizer(
        arm_lengths=(1, 1, 1),
        angles_sequence=test_angles,
        angle_mode='absolute',
        xy_sequence=xy_sequence
    )
    
    # 绘制关键帧
vis.plot_configuration(0)  # 第一帧
plt.show()
    
#vis.plot_configuration(-1) # 最后一帧
#plt.show()
    
    # 生成动画（实时预览）
vis.create_animation(interval=300, save_path="absolute_angle_arm.gif")