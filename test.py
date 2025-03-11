from traj_vis import *

data_str = """
Step 0: Theta1=0.000, Theta2=0.000, Theta3=0.000, at x=3, y =0
Step 1: Theta1=0.017, Theta2=0.016, Theta3=0.019, at x=3.0, y =0.13982047140598297
Step 2: Theta1=0.023, Theta2=0.025, Theta3=0.025, at x=2.91295712441206, y =0.2615903168916702
Step 3: Theta1=0.042, Theta2=0.042, Theta3=0.045, at x=2.8882374465465546, y =0.49533987045288086
Step 4: Theta1=0.009, Theta2=0.018, Theta3=0.008, at x=2.7067012786865234, y =0.41519858688116074
Step 5: Theta1=0.007, Theta2=0.022, Theta3=0.014, at x=2.4713948369026184, y =0.4988664761185646
Step 6: Theta1=0.010, Theta2=0.024, Theta3=0.018, at x=2.3564424961805344, y =0.557341955602169
Step 7: Theta1=0.019, Theta2=0.033, Theta3=0.025, at x=2.33413827419281, y =0.6176725029945374
Step 8: Theta1=-0.005, Theta2=0.016, Theta3=0.005, at x=2.2145859003067017, y =0.5796886906027794
Step 9: Theta1=-0.046, Theta2=-0.014, Theta3=-0.030, at x=2.053050309419632, y =0.39867305010557175
Step 10: Theta1=-0.002, Theta2=0.022, Theta3=0.007, at x=2.1182143166661263, y =0.45406991615891457
Step 11: Theta1=0.023, Theta2=0.043, Theta3=0.034, at x=2.0923560112714767, y =0.6042928881943226
Step 12: Theta1=-0.001, Theta2=0.027, Theta3=0.012, at x=2.039824418723583, y =0.5902112163603306
Step 13: Theta1=0.002, Theta2=0.042, Theta3=0.025, at x=1.828493706882, y =0.7015440054237843
Step 14: Theta1=0.019, Theta2=0.063, Theta3=0.035, at x=1.796068374067545, y =0.8278038688004017
Step 15: Theta1=0.033, Theta2=0.082, Theta3=0.046, at x=1.6930160410702229, y =1.0031594224274158
Step 16: Theta1=0.006, Theta2=0.091, Theta3=0.040, at x=1.527842152863741, y =1.0905025489628315
Step 17: Theta1=-0.049, Theta2=0.111, Theta3=0.035, at x=1.3386920429766178, y =1.06936851516366
Step 18: Theta1=-0.067, Theta2=0.117, Theta3=0.010, at x=1.3908666335046291, y =1.0572347156703472
Step 19: Theta1=-0.036, Theta2=0.144, Theta3=0.029, at x=1.1982829980552197, y =1.2073112837970257
Step 20: Theta1=-0.039, Theta2=0.158, Theta3=0.013, at x=1.0725125186145306, y =1.4022442512214184
Step 21: Theta1=-0.085, Theta2=0.141, Theta3=-0.047, at x=1.1367626003921032, y =1.5891385190188885
Step 22: Theta1=-0.094, Theta2=0.119, Theta3=-0.095, at x=1.2180305682122707, y =1.81984568759799
Step 23: Theta1=-0.099, Theta2=0.180, Theta3=-0.068, at x=1.2575333304703236, y =1.792474139481783
Step 24: Theta1=-0.087, Theta2=0.197, Theta3=-0.048, at x=1.2736591137945652, y =1.7957693003118038
Step 25: Theta1=-0.088, Theta2=0.186, Theta3=-0.055, at x=1.4168905951082706, y =1.7272192277014256
Step 26: Theta1=-0.050, Theta2=0.199, Theta3=-0.009, at x=1.3010858036577702, y =1.7966224886476994
Step 27: Theta1=-0.085, Theta2=0.295, Theta3=0.084, at x=1.0246956385672092, y =1.6889615096151829
Step 28: Theta1=-0.089, Theta2=0.310, Theta3=0.087, at x=0.7742901481688023, y =1.7619209177792072
Step 29: Theta1=-0.165, Theta2=0.290, Theta3=0.011, at x=0.7902133725583553, y =1.793625421822071
Step 30: Theta1=-0.160, Theta2=0.331, Theta3=0.020, at x=0.593865480273962, y =1.8903265669941902
Step 31: Theta1=-0.181, Theta2=0.548, Theta3=0.163, at x=0.44169218465685844, y =1.7098179534077644
Step 32: Theta1=-0.264, Theta2=0.374, Theta3=0.017, at x=0.37707624956965446, y =1.846591018140316
Step 33: Theta1=-0.348, Theta2=0.318, Theta3=-0.094, at x=0.3639608360826969, y =1.9624257758259773
Step 34: Theta1=-0.378, Theta2=0.253, Theta3=-0.215, at x=0.23711412027478218, y =2.208504967391491
Step 35: Theta1=-0.442, Theta2=0.286, Theta3=-0.277, at x=0.25206879526376724, y =2.2930566519498825
Step 36: Theta1=-0.497, Theta2=0.174, Theta3=-0.417, at x=0.2707025296986103, y =2.5054014176130295
Step 37: Theta1=-0.491, Theta2=0.217, Theta3=-0.437, at x=0.3362795375287533, y =2.5567684397101402
Step 38: Theta1=-0.403, Theta2=0.298, Theta3=-0.375, at x=0.22479327395558357, y =2.6609740927815437
Step 39: Theta1=-0.338, Theta2=0.530, Theta3=-0.200, at x=0.0885077603161335, y =2.591836243867874
Step 40: Theta1=-0.407, Theta2=0.431, Theta3=-0.259, at x=0.09083528816699982, y =2.6272447034716606
Step 41: Theta1=-0.467, Theta2=0.339, Theta3=-0.352, at x=0.12467744573950768, y =2.694973275065422
Step 42: Theta1=-0.472, Theta2=0.348, Theta3=-0.378, at x=0.15423398837447166, y =2.709876984357834
Step 43: Theta1=-0.477, Theta2=0.287, Theta3=-0.429, at x=0.1613619439303875, y =2.7968395575881004
Step 44: Theta1=-0.520, Theta2=0.133, Theta3=-0.549, at x=0.16314921528100967, y =3.0
Step 45: Theta1=-0.461, Theta2=0.277, Theta3=-0.480, at x=0.12265825271606445, y =3.0
Step 46: Theta1=-0.459, Theta2=0.314, Theta3=-0.467, at x=0.12899626418948174, y =3.0
Step 47: Theta1=-0.442, Theta2=0.348, Theta3=-0.440, at x=0.1113390177488327, y =3.0
Step 48: Theta1=-0.452, Theta2=0.339, Theta3=-0.447, at x=0.12705807387828827, y =3.0
Step 49: Theta1=-0.393, Theta2=0.540, Theta3=-0.297, at x=0.10975305736064911, y =2.8282757699489594
Step 50: Theta1=-0.349, Theta2=0.568, Theta3=-0.199, at x=0.006288416683673859, y =2.771343059837818
Step 51: Theta1=-0.396, Theta2=0.485, Theta3=-0.239, at x=-0.02896677702665329, y =2.7790516167879105
Step 52: Theta1=-0.421, Theta2=0.400, Theta3=-0.311, at x=-0.16175169497728348, y =2.9078036546707153
Step 53: Theta1=-0.547, Theta2=0.284, Theta3=-0.473, at x=-0.1203496977686882, y =3.0
Step 54: Theta1=-0.473, Theta2=0.446, Theta3=-0.386, at x=-0.23933103680610657, y =3.0
Step 55: Theta1=-0.501, Theta2=0.455, Theta3=-0.398, at x=-0.27242104709148407, y =3.0
Step 56: Theta1=-0.614, Theta2=0.299, Theta3=-0.540, at x=-0.09240315854549408, y =3.0
Step 57: Theta1=-0.545, Theta2=0.376, Theta3=-0.491, at x=-0.07336559891700745, y =2.9615870118141174
Step 58: Theta1=-0.481, Theta2=0.414, Theta3=-0.434, at x=-0.16651026904582977, y =3.0
Step 59: Theta1=-0.386, Theta2=0.581, Theta3=-0.260, at x=-0.4210731238126755, y =3.0
Step 60: Theta1=-0.386, Theta2=0.683, Theta3=-0.124, at x=-0.5891911536455154, y =2.9127214923501015
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