from FCNN_PC import FCNN_PC
from net_new import *
import torch
from traj_vis import *
import numpy as np
import matplotlib.pyplot as plt
from FK import FK


model = ResNetFCN(input_size=7, output_size=3, dropout_rate=0.1)
model.load_state_dict(torch.load("best_res_model.pth"))
model.eval()

lengths = np.array([1.0, 1.0, 1.0])
prev_angles = np.array([0.0, 0.0, 0.0]) # degrees
prev_position = np.array(FK(lengths, np.deg2rad(prev_angles)))
prev_angles /= 180
current_position = np.array([-2, 1])

input = torch.tensor(np.concatenate([current_position, prev_position, prev_angles]), dtype=torch.float32).unsqueeze(0)
#input = torch.tensor(current_position, dtype=torch.float32).unsqueeze(0)
angles = model(input).detach().numpy()[0] # -1 to 1
angles *= 180
print(angles)
print(FK(lengths, np.deg2rad(angles)))
vis = ArmVisualizer((1, 1, 1), [angles])

ax = vis.plot_configuration(0)
plt.show()
