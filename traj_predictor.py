'''
Part II 
Given a precise joint angle predictor, let's see how to predict a trajectory

'''
from net_new import *
from FCNN_PC import FCNN_PC
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from RL import *

if torch.cuda.is_available():
    device=torch.device("cuda")
    print("CUDA Activated!")
else:
    device=torch.device("cpu")

#model_pred = ResNetFCN(input_size=7, output_size=3, dropout_rate=0.1).to(device)
#model_pred.load_state_dict(torch.load("best_res_model.pth"))
model_pred = FCNN_PC(2).to(device)
model_pred.load_state_dict(torch.load("best_only_position_FCNN_PC_model.pth"))
model_pred.eval()
'''
def predict_trajectory(startx,starty,endx,endy,prevx_s,prevy_s,theta1,theta2,theta3):
    model = ResNetFCN(input_size=7, output_size=3, dropout_rate=0.1).to(device)
    model.load_state_dict(torch.load("best_res_model.pth"))
    model.eval()
    
def predict_first_point(x,y,model):
    predict_trajectory(3,0,x,y,3,0,0,0,0)
'''

model = LSTMPPOPolicy().to(device)
#model_1 = EnhancedLSTMPPO().to(device)
env = RobotTrajectoryEnv(model_pred, (3, 0),(0,0,0),(-3,0),device,max_steps=40)
print(device)
trajectories = train_rl(model, env,device=device,num_episodes=400)

# for i, (episode_reward, episode_traj) in enumerate(trajectories):
#     if episode_reward > best_reward:
#         best_reward = episode_reward
#         best_idx = i
# Print the trajectory for the last episode
for t, (x,y,theta1, theta2, theta3) in enumerate(trajectories[-1][1]):
    print(f"Step {t}: Theta1={theta1:.3f}, Theta2={theta2:.3f}, Theta3={theta3:.3f}, at x={x}, y ={y}")


    
    