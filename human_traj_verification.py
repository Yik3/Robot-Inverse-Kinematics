from net_new import *
import torch
import torch.nn as nn
from traj_vis import *
from matplotlib import pyplot as plt

if torch.cuda.is_available():
    device=torch.device("cuda")
    print("CUDA Activated!")
else:
    device=torch.device("cpu")
    
model = Transformer().to(device)

model.load_state_dict(torch.load("best_trans_model.pth"))
model.eval()

import numpy as np

def human_gen_traj(start, end, sample):
    if sample < 0:
        raise ValueError("???")
    
    if (start[0]**2+start[1]**2)>9:
        raise ValueError("???")

    if (end[0]**2+end[1]**2)>9:
        raise ValueError("???")
    x_vals = np.linspace(start[0], end[0], sample+2)
    y_vals = np.linspace(start[1], end[1], sample+2)

    return [(x, y) for x, y in zip(x_vals, y_vals)]

start_point = (3, 0)
end_point = (0,-3)
samples = 15
path = human_gen_traj(start_point, end_point, samples)
#print(trajectory)

def joint_traj(trajectory,startangles,device,model):
    joint_traj = []
    start = trajectory[0]
    end = trajectory[-1]
    prevx = start[0]
    prevy = start[1]
    ptheta1 = startangles[0]
    ptheta2 = startangles[1]
    ptheta3 = startangles[2]
    joint_traj.append((prevx,prevy,ptheta1,ptheta2,ptheta3))
    for point in trajectory[1:]:
        new_x = point[0]
        new_y = point[1]
        input_tensor = torch.tensor([new_x, new_y, prevx, prevy, ptheta1, ptheta2, ptheta3], dtype=torch.float32).unsqueeze(0).to(device)
        new_theta1, new_theta2, new_theta3 = model(input_tensor).detach().cpu().numpy().squeeze()
        joint_traj.append((new_x,new_y,new_theta1,new_theta2,new_theta3))
        prevx = new_x
        prevy = new_y
        ptheta1 = new_theta1
        ptheta2 = new_theta2
        ptheta3 = new_theta3
    return joint_traj

trajectories = joint_traj(path,(0,0,0),device,model)
outstr = ""
for t, (x,y,theta1, theta2, theta3) in enumerate(trajectories):
    print(f"Step {t}: Theta1={theta1:.3f}, Theta2={theta2:.3f}, Theta3={theta3:.3f}, at x={x}, y ={y}")
    outstr += f"Step {t}: Theta1={theta1:.3f}, Theta2={theta2:.3f}, Theta3={theta3:.3f}, at x={x}, y ={y} \n"
    
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
theta_sequence,xy_sequence = parse_angle_data(outstr)
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
#vis.plot_configuration(0)  # 第一帧
#plt.show()
    
#vis.plot_configuration(-1) # 最后一帧
#plt.show()

vis.create_animation(interval=300, save_path="test.gif")