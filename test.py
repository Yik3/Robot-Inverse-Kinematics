from traj_vis import *

data_str = """
Step 0: Theta1=0.000, Theta2=0.000, Theta3=0.000
Step 1: Theta1=0.016, Theta2=0.023, Theta3=-0.001
Step 2: Theta1=0.007, Theta2=0.025, Theta3=-0.016
Step 3: Theta1=-0.007, Theta2=0.026, Theta3=-0.016
Step 4: Theta1=-0.040, Theta2=-0.009, Theta3=-0.036
Step 5: Theta1=-0.031, Theta2=0.001, Theta3=-0.015
Step 6: Theta1=0.002, Theta2=0.040, Theta3=0.023
Step 7: Theta1=-0.002, Theta2=0.046, Theta3=0.039
Step 8: Theta1=-0.023, Theta2=0.041, Theta3=0.043
Step 9: Theta1=-0.029, Theta2=0.039, Theta3=0.049
Step 10: Theta1=-0.061, Theta2=0.051, Theta3=0.053
Step 11: Theta1=-0.096, Theta2=0.003, Theta3=0.007
Step 12: Theta1=-0.101, Theta2=0.038, Theta3=0.041
Step 13: Theta1=-0.105, Theta2=0.082, Theta3=0.091
Step 14: Theta1=-0.109, Theta2=0.069, Theta3=0.065
Step 15: Theta1=-0.080, Theta2=-0.014, Theta3=-0.016
Step 16: Theta1=-0.069, Theta2=0.025, Theta3=0.033
Step 17: Theta1=-0.063, Theta2=0.055, Theta3=0.079
Step 18: Theta1=-0.079, Theta2=0.083, Theta3=0.116
Step 19: Theta1=-0.083, Theta2=0.085, Theta3=0.108
Step 20: Theta1=-0.086, Theta2=0.089, Theta3=0.128
Step 21: Theta1=-0.100, Theta2=0.080, Theta3=0.101
Step 22: Theta1=-0.099, Theta2=0.094, Theta3=0.115
Step 23: Theta1=-0.084, Theta2=0.044, Theta3=0.050
Step 24: Theta1=-0.089, Theta2=0.064, Theta3=0.074
Step 25: Theta1=-0.076, Theta2=0.073, Theta3=0.089
Step 26: Theta1=-0.092, Theta2=0.102, Theta3=0.131
Step 27: Theta1=-0.127, Theta2=0.074, Theta3=0.088
Step 28: Theta1=-0.125, Theta2=0.097, Theta3=0.109
Step 29: Theta1=-0.124, Theta2=0.110, Theta3=0.120
Step 30: Theta1=-0.127, Theta2=0.064, Theta3=0.067
Step 31: Theta1=-0.114, Theta2=0.096, Theta3=0.098
Step 32: Theta1=-0.100, Theta2=0.076, Theta3=0.079
Step 33: Theta1=-0.048, Theta2=0.024, Theta3=0.020
Step 34: Theta1=-0.047, Theta2=0.048, Theta3=0.048
Step 35: Theta1=-0.059, Theta2=0.066, Theta3=0.072
Step 36: Theta1=-0.077, Theta2=0.091, Theta3=0.108
Step 37: Theta1=-0.092, Theta2=0.099, Theta3=0.116
Step 38: Theta1=-0.117, Theta2=0.067, Theta3=0.072
Step 39: Theta1=-0.120, Theta2=0.093, Theta3=0.103
Step 40: Theta1=-0.130, Theta2=0.039, Theta3=0.031
Step 41: Theta1=-0.141, Theta2=0.055, Theta3=0.054
Step 42: Theta1=-0.129, Theta2=0.056, Theta3=0.063
Step 43: Theta1=-0.105, Theta2=0.089, Theta3=0.123
Step 44: Theta1=-0.103, Theta2=0.120, Theta3=0.161
Step 45: Theta1=-0.111, Theta2=0.159, Theta3=0.188
Step 46: Theta1=-0.154, Theta2=0.097, Theta3=0.098
Step 47: Theta1=-0.166, Theta2=0.131, Theta3=0.132
Step 48: Theta1=-0.171, Theta2=0.131, Theta3=0.132
Step 49: Theta1=-0.174, Theta2=0.073, Theta3=0.075
Step 50: Theta1=-0.144, Theta2=0.104, Theta3=0.115
Step 51: Theta1=-0.133, Theta2=0.088, Theta3=0.094
Step 52: Theta1=-0.117, Theta2=0.124, Theta3=0.149
Step 53: Theta1=-0.145, Theta2=0.105, Theta3=0.113
Step 54: Theta1=-0.150, Theta2=0.070, Theta3=0.081
Step 55: Theta1=-0.127, Theta2=0.090, Theta3=0.103
Step 56: Theta1=-0.132, Theta2=0.100, Theta3=0.110
Step 57: Theta1=-0.131, Theta2=0.061, Theta3=0.078
Step 58: Theta1=-0.107, Theta2=0.076, Theta3=0.095
Step 59: Theta1=-0.087, Theta2=0.065, Theta3=0.081
Step 60: Theta1=-0.047, Theta2=0.068, Theta3=0.075
Step 61: Theta1=-0.046, Theta2=0.078, Theta3=0.091
Step 62: Theta1=-0.083, Theta2=0.068, Theta3=0.072
Step 63: Theta1=-0.104, Theta2=0.082, Theta3=0.089
Step 64: Theta1=-0.106, Theta2=0.083, Theta3=0.091
Step 65: Theta1=-0.080, Theta2=0.054, Theta3=0.062
Step 66: Theta1=-0.077, Theta2=0.064, Theta3=0.090
Step 67: Theta1=-0.069, Theta2=0.078, Theta3=0.100
Step 68: Theta1=-0.083, Theta2=0.114, Theta3=0.137
Step 69: Theta1=-0.074, Theta2=0.055, Theta3=0.068
Step 70: Theta1=-0.071, Theta2=0.070, Theta3=0.085
Step 71: Theta1=-0.081, Theta2=0.084, Theta3=0.102
Step 72: Theta1=-0.074, Theta2=0.056, Theta3=0.058
Step 73: Theta1=-0.078, Theta2=0.071, Theta3=0.075
Step 74: Theta1=-0.112, Theta2=0.078, Theta3=0.083
Step 75: Theta1=-0.101, Theta2=0.136, Theta3=0.181
Step 76: Theta1=-0.122, Theta2=0.219, Theta3=0.263
Step 77: Theta1=-0.180, Theta2=0.229, Theta3=0.240
Step 78: Theta1=-0.153, Theta2=0.283, Theta3=0.291
Step 79: Theta1=-0.217, Theta2=0.197, Theta3=0.188
Step 80: Theta1=-0.195, Theta2=0.151, Theta3=0.147
Step 81: Theta1=-0.133, Theta2=0.086, Theta3=0.095
Step 82: Theta1=-0.108, Theta2=0.080, Theta3=0.100
Step 83: Theta1=-0.095, Theta2=0.083, Theta3=0.098
Step 84: Theta1=-0.087, Theta2=0.060, Theta3=0.077
Step 85: Theta1=-0.060, Theta2=0.054, Theta3=0.067
Step 86: Theta1=-0.011, Theta2=0.059, Theta3=0.067
Step 87: Theta1=-0.021, Theta2=0.050, Theta3=0.063
Step 88: Theta1=-0.052, Theta2=0.047, Theta3=0.065
Step 89: Theta1=-0.046, Theta2=0.061, Theta3=0.070
Step 90: Theta1=-0.074, Theta2=0.082, Theta3=0.091
Step 91: Theta1=-0.071, Theta2=0.063, Theta3=0.068
Step 92: Theta1=-0.076, Theta2=0.071, Theta3=0.079
Step 93: Theta1=-0.078, Theta2=0.072, Theta3=0.080
Step 94: Theta1=-0.081, Theta2=0.060, Theta3=0.068
Step 95: Theta1=-0.075, Theta2=0.056, Theta3=0.074
Step 96: Theta1=-0.076, Theta2=0.049, Theta3=0.062
Step 97: Theta1=-0.079, Theta2=0.060, Theta3=0.069
Step 98: Theta1=-0.088, Theta2=0.052, Theta3=0.070
Step 99: Theta1=-0.091, Theta2=0.061, Theta3=0.075
Step 100: Theta1=-0.084, Theta2=0.048, Theta3=0.053
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
print(theta_sequence[99])

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