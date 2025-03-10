from traj_vis import *

data_str = """
Step 0: Theta1=0.000, Theta2=0.000, Theta3=0.000
Step 1: Theta1=0.016, Theta2=0.022, Theta3=0.021
Step 2: Theta1=0.022, Theta2=0.024, Theta3=0.028
Step 3: Theta1=0.012, Theta2=0.015, Theta3=0.015
Step 4: Theta1=0.018, Theta2=0.019, Theta3=0.020
Step 5: Theta1=0.022, Theta2=0.024, Theta3=0.024
Step 6: Theta1=0.018, Theta2=0.025, Theta3=0.023
Step 7: Theta1=0.029, Theta2=0.034, Theta3=0.032
Step 8: Theta1=0.046, Theta2=0.050, Theta3=0.048
Step 9: Theta1=0.022, Theta2=0.036, Theta3=0.026
Step 10: Theta1=0.021, Theta2=0.034, Theta3=0.020
Step 11: Theta1=0.019, Theta2=0.037, Theta3=0.026
Step 12: Theta1=0.029, Theta2=0.046, Theta3=0.029
Step 13: Theta1=0.029, Theta2=0.046, Theta3=0.034
Step 14: Theta1=0.058, Theta2=0.074, Theta3=0.060
Step 15: Theta1=0.052, Theta2=0.073, Theta3=0.047
Step 16: Theta1=0.043, Theta2=0.066, Theta3=0.037
Step 17: Theta1=0.014, Theta2=0.045, Theta3=0.016
Step 18: Theta1=0.030, Theta2=0.058, Theta3=0.028
Step 19: Theta1=0.028, Theta2=0.064, Theta3=0.039
Step 20: Theta1=0.029, Theta2=0.066, Theta3=0.029
Step 21: Theta1=0.027, Theta2=0.062, Theta3=0.025
Step 22: Theta1=0.023, Theta2=0.059, Theta3=0.026
Step 23: Theta1=0.046, Theta2=0.078, Theta3=0.045
Step 24: Theta1=0.039, Theta2=0.075, Theta3=0.038
Step 25: Theta1=0.016, Theta2=0.060, Theta3=0.019
Step 26: Theta1=-0.003, Theta2=0.043, Theta3=0.002
Step 27: Theta1=0.016, Theta2=0.054, Theta3=0.015
Step 28: Theta1=0.063, Theta2=0.089, Theta3=0.058
Step 29: Theta1=0.075, Theta2=0.101, Theta3=0.068
Step 30: Theta1=0.053, Theta2=0.089, Theta3=0.046
Step 31: Theta1=-0.009, Theta2=0.050, Theta3=0.005
Step 32: Theta1=0.046, Theta2=0.088, Theta3=0.039
Step 33: Theta1=0.059, Theta2=0.098, Theta3=0.049
Step 34: Theta1=0.077, Theta2=0.111, Theta3=0.064
Step 35: Theta1=0.037, Theta2=0.099, Theta3=0.045
Step 36: Theta1=0.046, Theta2=0.105, Theta3=0.034
Step 37: Theta1=0.035, Theta2=0.096, Theta3=0.021
Step 38: Theta1=-0.018, Theta2=0.076, Theta3=0.010
Step 39: Theta1=-0.022, Theta2=0.066, Theta3=-0.000
Step 40: Theta1=0.030, Theta2=0.092, Theta3=0.032
Step 41: Theta1=0.043, Theta2=0.098, Theta3=0.032
Step 42: Theta1=0.062, Theta2=0.109, Theta3=0.052
Step 43: Theta1=0.049, Theta2=0.104, Theta3=0.036
Step 44: Theta1=0.013, Theta2=0.078, Theta3=0.009
Step 45: Theta1=0.049, Theta2=0.096, Theta3=0.033
Step 46: Theta1=0.065, Theta2=0.104, Theta3=0.049
Step 47: Theta1=0.067, Theta2=0.105, Theta3=0.047
Step 48: Theta1=0.066, Theta2=0.104, Theta3=0.044
Step 49: Theta1=0.086, Theta2=0.111, Theta3=0.060
Step 50: Theta1=0.061, Theta2=0.094, Theta3=0.032
Step 51: Theta1=0.050, Theta2=0.083, Theta3=0.018
Step 52: Theta1=0.048, Theta2=0.088, Theta3=0.019
Step 53: Theta1=0.052, Theta2=0.070, Theta3=0.023
Step 54: Theta1=0.020, Theta2=0.009, Theta3=0.003
Step 55: Theta1=0.018, Theta2=0.020, Theta3=-0.012
Step 56: Theta1=0.024, Theta2=0.043, Theta3=-0.008
Step 57: Theta1=0.052, Theta2=0.063, Theta3=0.022
Step 58: Theta1=0.054, Theta2=0.078, Theta3=0.027
Step 59: Theta1=0.031, Theta2=0.077, Theta3=0.015
Step 60: Theta1=0.047, Theta2=0.091, Theta3=0.032
Step 61: Theta1=0.048, Theta2=0.092, Theta3=0.030
Step 62: Theta1=0.047, Theta2=0.090, Theta3=0.028
Step 63: Theta1=0.023, Theta2=0.072, Theta3=0.012
Step 64: Theta1=0.041, Theta2=0.081, Theta3=0.025
Step 65: Theta1=0.033, Theta2=0.070, Theta3=0.018
Step 66: Theta1=0.047, Theta2=0.077, Theta3=0.035
Step 67: Theta1=0.072, Theta2=0.093, Theta3=0.059
Step 68: Theta1=0.047, Theta2=0.078, Theta3=0.036
Step 69: Theta1=0.063, Theta2=0.090, Theta3=0.049
Step 70: Theta1=0.031, Theta2=0.072, Theta3=0.026
"""

# 数据转换函数
def parse_angle_data(data_string):
    angle_sequence = []
    
    # 按行分割数据
    lines = [line.strip() for line in data_string.split('\n') if line.strip()]
    
    for line in lines:
        # 提取数值部分
        values_part = line.split(": ")[1]  # 获取"Theta1=0.000, Theta2=0.000, Theta3=0.000"
        
        # 分割三个角度值
        theta_strs = values_part.split(", ")
        
        # 提取每个角度值
        angles = []
        for t_str in theta_strs:
            # 使用split('=')分割并取第二个元素
            angle_value = float(t_str.split('=')[1])*180.0
            angles.append(angle_value)
        
        # 转换为元组并添加到序列
        angle_sequence.append(tuple(angles))
    
    return angle_sequence

# 执行转换
theta_sequence = parse_angle_data(data_str)
print(theta_sequence[-1])

test_angles = theta_sequence
    
    # 初始化可视化工具（使用绝对角度模式）
vis = ArmVisualizer(
        arm_lengths=(1, 1, 1),
        angles_sequence=test_angles,
        angle_mode='absolute'
    )
    
    # 绘制关键帧
vis.plot_configuration(0)  # 第一帧
plt.show()
    
vis.plot_configuration(-1) # 最后一帧
plt.show()
    
    # 生成动画（实时预览）
vis.create_animation(interval=300, save_path="absolute_angle_arm.gif")