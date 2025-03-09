from traj_vis import *

data_str = """
Step 0: Theta1=0.000, Theta2=0.000, Theta3=0.000
Step 1: Theta1=0.013, Theta2=0.020, Theta3=-0.004
Step 2: Theta1=0.013, Theta2=0.033, Theta3=-0.009
Step 3: Theta1=-0.004, Theta2=0.029, Theta3=-0.009
Step 4: Theta1=-0.057, Theta2=-0.025, Theta3=-0.043
Step 5: Theta1=-0.099, Theta2=-0.084, Theta3=-0.075
Step 6: Theta1=-0.126, Theta2=-0.127, Theta3=-0.082
Step 7: Theta1=-0.128, Theta2=-0.083, Theta3=0.056
Step 8: Theta1=-0.044, Theta2=0.167, Theta3=0.397
Step 9: Theta1=0.164, Theta2=0.487, Theta3=0.687
Step 10: Theta1=-0.024, Theta2=0.574, Theta3=0.680
Step 11: Theta1=-0.366, Theta2=0.587, Theta3=0.545
Step 12: Theta1=-0.225, Theta2=0.779, Theta3=0.690
Step 13: Theta1=-0.169, Theta2=0.828, Theta3=0.732
Step 14: Theta1=-0.148, Theta2=0.839, Theta3=0.743
Step 15: Theta1=-0.140, Theta2=0.842, Theta3=0.746
Step 16: Theta1=-0.137, Theta2=0.843, Theta3=0.746
Step 17: Theta1=-0.136, Theta2=0.843, Theta3=0.746
Step 18: Theta1=-0.136, Theta2=0.843, Theta3=0.746
Step 19: Theta1=-0.503, Theta2=0.716, Theta3=0.497
Step 20: Theta1=-0.367, Theta2=0.769, Theta3=0.633
Step 21: Theta1=-0.164, Theta2=0.832, Theta3=0.756
Step 22: Theta1=-0.158, Theta2=0.839, Theta3=0.756
Step 23: Theta1=-0.062, Theta2=0.858, Theta3=0.790
Step 24: Theta1=-0.095, Theta2=0.852, Theta3=0.769
Step 25: Theta1=-0.533, Theta2=0.698, Theta3=0.470
Step 26: Theta1=-0.429, Theta2=0.745, Theta3=0.596
Step 27: Theta1=-0.476, Theta2=0.734, Theta3=0.583
Step 28: Theta1=-0.099, Theta2=0.842, Theta3=0.793
Step 29: Theta1=-0.262, Theta2=0.815, Theta3=0.715
Step 30: Theta1=-0.084, Theta2=0.854, Theta3=0.798
Step 31: Theta1=-0.230, Theta2=0.824, Theta3=0.727
Step 32: Theta1=-0.511, Theta2=0.723, Theta3=0.544
Step 33: Theta1=-0.435, Theta2=0.751, Theta3=0.623
Step 34: Theta1=-0.296, Theta2=0.802, Theta3=0.720
Step 35: Theta1=-0.608, Theta2=0.664, Theta3=0.467
Step 36: Theta1=-0.734, Theta2=0.527, Theta3=0.264
Step 37: Theta1=-0.422, Theta2=0.723, Theta3=0.621
Step 38: Theta1=-0.445, Theta2=0.754, Theta3=0.657
Step 39: Theta1=-0.795, Theta2=0.437, Theta3=0.162
Step 40: Theta1=-0.175, Theta2=0.781, Theta3=0.750
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