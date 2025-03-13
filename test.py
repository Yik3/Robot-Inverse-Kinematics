from traj_vis import *

data_str = """
Step 0: Theta1=0.000, Theta2=0.000, Theta3=0.000
Step 1: Theta1=0.026, Theta2=0.025, Theta3=0.028
Step 2: Theta1=0.029, Theta2=0.029, Theta3=0.033
Step 3: Theta1=0.052, Theta2=0.051, Theta3=0.055
Step 4: Theta1=0.075, Theta2=0.075, Theta3=0.086
Step 5: Theta1=0.117, Theta2=0.134, Theta3=0.150
Step 6: Theta1=0.154, Theta2=0.184, Theta3=0.179
Step 7: Theta1=0.162, Theta2=0.183, Theta3=0.167
Step 8: Theta1=0.182, Theta2=0.224, Theta3=0.175
Step 9: Theta1=0.182, Theta2=0.259, Theta3=0.195
Step 10: Theta1=0.186, Theta2=0.261, Theta3=0.190
Step 11: Theta1=0.192, Theta2=0.253, Theta3=0.146
Step 12: Theta1=0.191, Theta2=0.259, Theta3=0.169
Step 13: Theta1=0.176, Theta2=0.249, Theta3=0.201
Step 14: Theta1=0.172, Theta2=0.263, Theta3=0.214
Step 15: Theta1=0.148, Theta2=0.249, Theta3=0.217
Step 16: Theta1=0.160, Theta2=0.264, Theta3=0.222
Step 17: Theta1=0.179, Theta2=0.268, Theta3=0.201
Step 18: Theta1=0.189, Theta2=0.256, Theta3=0.176
Step 19: Theta1=0.186, Theta2=0.261, Theta3=0.191
Step 20: Theta1=0.186, Theta2=0.275, Theta3=0.183
Step 21: Theta1=0.167, Theta2=0.295, Theta3=0.222
Step 22: Theta1=0.172, Theta2=0.236, Theta3=0.198
Step 23: Theta1=0.178, Theta2=0.251, Theta3=0.202
Step 24: Theta1=0.155, Theta2=0.278, Theta3=0.228
Step 25: Theta1=0.186, Theta2=0.246, Theta3=0.113
Step 26: Theta1=0.178, Theta2=0.251, Theta3=0.189
Step 27: Theta1=0.176, Theta2=0.290, Theta3=0.196
Step 28: Theta1=0.183, Theta2=0.286, Theta3=0.173
Step 29: Theta1=0.161, Theta2=0.241, Theta3=0.199
Step 30: Theta1=0.183, Theta2=0.257, Theta3=0.184
Step 31: Theta1=0.186, Theta2=0.256, Theta3=0.187
Step 32: Theta1=0.173, Theta2=0.290, Theta3=0.213
Step 33: Theta1=0.182, Theta2=0.277, Theta3=0.197
Step 34: Theta1=0.185, Theta2=0.271, Theta3=0.191
Step 35: Theta1=0.168, Theta2=0.213, Theta3=0.182
Step 36: Theta1=0.186, Theta2=0.230, Theta3=0.168
Step 37: Theta1=0.171, Theta2=0.255, Theta3=0.209
Step 38: Theta1=0.162, Theta2=0.260, Theta3=0.220
Step 39: Theta1=0.181, Theta2=0.265, Theta3=0.195
Step 40: Theta1=0.184, Theta2=0.251, Theta3=0.192
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