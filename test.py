from traj_vis import *

data_str = """
Step 0: Theta1=0.000, Theta2=0.000, Theta3=0.000, at x=3, y =0
Step 1: Theta1=-0.103, Theta2=0.075, Theta3=0.055, at x=2.990414619445801, y =0.027836844325065613
Step 2: Theta1=-0.097, Theta2=0.076, Theta3=0.068, at x=2.990107238292694, y =0.04692472517490387
Step 3: Theta1=-0.110, Theta2=0.071, Theta3=0.032, at x=2.9910058975219727, y =-0.007038000971078873
Step 4: Theta1=-0.092, Theta2=0.101, Theta3=0.101, at x=2.985555112361908, y =0.11037809401750565
Step 5: Theta1=-0.084, Theta2=0.076, Theta3=0.090, at x=2.9895421266555786, y =0.08095691353082657
Step 6: Theta1=-0.102, Theta2=0.075, Theta3=0.058, at x=2.990356147289276, y =0.03150011971592903
Step 7: Theta1=-0.108, Theta2=0.073, Theta3=0.041, at x=2.990736186504364, y =0.0063268691301345825
Step 8: Theta1=-0.124, Theta2=0.106, Theta3=0.050, at x=2.9854716658592224, y =0.03238014504313469
Step 9: Theta1=-0.133, Theta2=0.111, Theta3=0.031, at x=2.984515368938446, y =0.009048746898770332
Step 10: Theta1=-0.142, Theta2=0.144, Theta3=0.055, at x=2.9780808091163635, y =0.05615496635437012
Step 11: Theta1=-0.139, Theta2=0.115, Theta3=0.016, at x=2.983577251434326, y =-0.00791170634329319
Step 12: Theta1=-0.141, Theta2=0.128, Theta3=0.034, at x=2.981318175792694, y =0.021783754229545593
Step 13: Theta1=-0.129, Theta2=0.104, Theta3=0.033, at x=2.985750377178192, y =0.008370425552129745
Step 14: Theta1=-0.147, Theta2=0.140, Theta3=0.034, at x=2.9788278937339783, y =0.026734113693237305
Step 15: Theta1=-0.111, Theta2=0.068, Theta3=0.025, at x=2.9912302494049072, y =-0.01745997928082943
Step 16: Theta1=-0.109, Theta2=0.072, Theta3=0.036, at x=2.9908562898635864, y =-0.00017426535487174988
Step 17: Theta1=-0.137, Theta2=0.125, Theta3=0.040, at x=2.982031762599945, y =0.027466394007205963
Step 18: Theta1=-0.140, Theta2=0.118, Theta3=0.018, at x=2.9831076860427856, y =-0.004137024283409119
Step 19: Theta1=-0.106, Theta2=0.074, Theta3=0.046, at x=2.990637183189392, y =0.013420511037111282
Step 20: Theta1=-0.106, Theta2=0.074, Theta3=0.047, at x=2.990536868572235, y =0.01608554646372795
Step 21: Theta1=-0.110, Theta2=0.069, Theta3=0.028, at x=2.9911402463912964, y =-0.012975392863154411
Step 22: Theta1=-0.127, Theta2=0.105, Theta3=0.039, at x=2.985735595226288, y =0.01676369085907936
Step 23: Theta1=-0.142, Theta2=0.138, Theta3=0.045, at x=2.979338228702545, y =0.04050011187791824
Step 24: Theta1=-0.112, Theta2=0.075, Theta3=0.031, at x=2.9904359579086304, y =-0.006679253652691841
Step 25: Theta1=-0.104, Theta2=0.074, Theta3=0.052, at x=2.990499436855316, y =0.022754184901714325
Step 26: Theta1=-0.142, Theta2=0.133, Theta3=0.038, at x=2.9804518818855286, y =0.02884386107325554
Step 27: Theta1=-0.102, Theta2=0.075, Theta3=0.056, at x=2.9903899431228638, y =0.029260709881782532
Step 28: Theta1=-0.103, Theta2=0.092, Theta3=0.077, at x=2.9875240325927734, y =0.06645287573337555
Step 29: Theta1=-0.104, Theta2=0.074, Theta3=0.051, at x=2.9905225038528442, y =0.02129524201154709
Step 30: Theta1=-0.143, Theta2=0.142, Theta3=0.048, at x=2.9786118865013123, y =0.046520911157131195
Step 31: Theta1=-0.151, Theta2=0.133, Theta3=0.004, at x=2.979699969291687, y =-0.013242118060588837
Step 32: Theta1=-0.130, Theta2=0.096, Theta3=0.013, at x=2.9868257641792297, y =-0.020679982379078865
Step 33: Theta1=-0.098, Theta2=0.076, Theta3=0.067, at x=2.9901428818702698, y =0.044885165989398956
Step 34: Theta1=-0.114, Theta2=0.096, Theta3=0.060, at x=2.9871695041656494, y =0.04228278249502182
Step 35: Theta1=-0.148, Theta2=0.145, Theta3=0.039, at x=2.9777756333351135, y =0.03660023957490921
Step 36: Theta1=-0.112, Theta2=0.064, Theta3=0.014, at x=2.9915425181388855, y =-0.03351742308586836
Step 37: Theta1=-0.089, Theta2=0.076, Theta3=0.083, at x=2.9896755814552307, y =0.06961295753717422
Step 38: Theta1=-0.128, Theta2=0.141, Theta3=0.082, at x=2.9785194993019104, y =0.09555394947528839
Step 39: Theta1=-0.108, Theta2=0.072, Theta3=0.038, at x=2.9908028841018677, y =0.0027146637439727783
Step 40: Theta1=-0.108, Theta2=0.089, Theta3=0.063, at x=2.988202452659607, y =0.04409743845462799
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