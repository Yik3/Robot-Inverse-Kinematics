'''
Part II 
Given a precise joint angle predictor, let's see how to predict a trajectory

'''
from net_new import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device=torch.device("cuda")
    print("CUDA Activated!")
else:
    device=torch.device("cpu")

model_pred = ResNetFCN(input_size=7, output_size=3, dropout_rate=0.1).to(device)
model_pred.load_state_dict(torch.load("best_res_model.pth"))
model_pred.eval()
'''
def predict_trajectory(startx,starty,endx,endy,prevx_s,prevy_s,theta1,theta2,theta3):
    model = ResNetFCN(input_size=7, output_size=3, dropout_rate=0.1).to(device)
    model.load_state_dict(torch.load("best_res_model.pth"))
    model.eval()
    
def predict_first_point(x,y,model):
    predict_trajectory(3,0,x,y,3,0,0,0,0)
'''


    
    