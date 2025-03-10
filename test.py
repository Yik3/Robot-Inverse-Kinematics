from traj_vis import *

data_str = """
Step 0: Theta1=0.000, Theta2=0.000, Theta3=0.000
Step 1: Theta1=-0.008, Theta2=-0.005, Theta3=-0.006
Step 2: Theta1=-0.039, Theta2=-0.031, Theta3=-0.038
Step 3: Theta1=-0.133, Theta2=-0.078, Theta3=-0.122
Step 4: Theta1=-0.136, Theta2=-0.017, Theta3=-0.134
Step 5: Theta1=0.311, Theta2=0.421, Theta3=0.196
Step 6: Theta1=0.868, Theta2=0.755, Theta3=0.702
Step 7: Theta1=0.987, Theta2=0.964, Theta3=0.939
Step 8: Theta1=0.996, Theta2=0.988, Theta3=0.977
Step 9: Theta1=0.996, Theta2=0.989, Theta3=0.980
Step 10: Theta1=0.996, Theta2=0.989, Theta3=0.980
Step 11: Theta1=0.996, Theta2=0.989, Theta3=0.980
Step 12: Theta1=0.996, Theta2=0.989, Theta3=0.980
Step 13: Theta1=0.996, Theta2=0.989, Theta3=0.980
Step 14: Theta1=0.996, Theta2=0.989, Theta3=0.980
Step 15: Theta1=0.996, Theta2=0.989, Theta3=0.980
Step 16: Theta1=0.996, Theta2=0.989, Theta3=0.980
Step 17: Theta1=0.996, Theta2=0.989, Theta3=0.980
Step 18: Theta1=0.996, Theta2=0.989, Theta3=0.980
Step 19: Theta1=0.996, Theta2=0.989, Theta3=0.980
Step 20: Theta1=0.996, Theta2=0.989, Theta3=0.980
Step 21: Theta1=0.996, Theta2=0.989, Theta3=0.980
Step 22: Theta1=0.996, Theta2=0.989, Theta3=0.980
Step 23: Theta1=0.996, Theta2=0.989, Theta3=0.980
Step 24: Theta1=0.996, Theta2=0.989, Theta3=0.980
Step 25: Theta1=0.996, Theta2=0.989, Theta3=0.980
Step 26: Theta1=0.996, Theta2=0.989, Theta3=0.980
Step 27: Theta1=0.996, Theta2=0.989, Theta3=0.980
Step 28: Theta1=0.996, Theta2=0.989, Theta3=0.980
Step 29: Theta1=0.996, Theta2=0.989, Theta3=0.980
Step 30: Theta1=0.996, Theta2=0.989, Theta3=0.980
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