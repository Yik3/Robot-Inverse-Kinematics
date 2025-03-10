import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gymnasium import spaces
import torch.nn.functional as F
import torch.distributions as distributions

'''
RL set up
'''
class RobotTrajectoryEnv(gym.Env):
    def __init__(self, predictor_model, start_pos, start_thetas, end_pos, device,max_steps=50):
        super(RobotTrajectoryEnv, self).__init__()

        self.predictor_model = predictor_model  # Your existing model
        self.start_pos = np.array(start_pos)  # (x_start, y_start)
        self.start_thetas = np.array(start_thetas)  # (theta1_start, theta2_start, theta3_start)
        self.end_pos = np.array(end_pos)  # (x_end, y_end)
        self.state = None  # (x, y, prev_x, prev_y, theta1, theta2, theta3)
        self.max_steps = max_steps
        self.current_step = 0
        self.device = device
        # Action space now modifies x, y (small changes)
        self.action_space = spaces.Box(low=-0.3, high=0.3, shape=(2,), dtype=np.float32)

        # Observation space includes theta1, theta2, theta3 from the start
        self.observation_space = spaces.Box(
            low=np.array([-3, -3, -3, -3, -1, -1, -1]),  # Min values
            high=np.array([3, 3, 3, 3, 1, 1, 1]),        # Max values
            dtype=np.float32
        )

    def reset(self):
        """Reset the environment to the initial state."""
        self.current_step = 0

        # Directly use the provided initial thetas instead of predicting them
        self.state = np.concatenate([self.start_pos, self.start_pos, self.start_thetas])
        return self.state

    def step(self, action):
        """Apply an action by moving (x, y) and predict the new joint angles."""
        self.current_step += 1
        prev_state = self.state.copy()
        #print(action)
    # Update x, y with small changes
        new_x = np.clip(prev_state[0] + action[0], -3, 3)
        new_y = np.clip(prev_state[1] + action[1], -3, 3)

    # Move input tensor to the same device as predictor_model
        input_tensor = torch.tensor([new_x, new_y, prev_state[0], prev_state[1], prev_state[4], prev_state[5], prev_state[6]], dtype=torch.float32).unsqueeze(0).to(self.device)
    
    # Ensure predictor model is on CUDA
        new_theta1, new_theta2, new_theta3 = self.predictor_model(input_tensor).detach().cpu().numpy().squeeze()

    # Compute reward
        dist_to_goal = np.linalg.norm([new_x, new_y] - self.end_pos)
        prev_dist = np.linalg.norm([prev_state[0], prev_state[1]] - self.end_pos)

        reward = 0
        reward = -dist_to_goal
        #reward = (prev_dist - dist_to_goal) * 2.9  # Reward for getting closer

    # Penalize large joint angle changes (minimizing θ changes for smoother motion)
        theta_change = np.linalg.norm([
            new_theta1 - prev_state[4],
            new_theta2 - prev_state[5],
            new_theta3 - prev_state[6]
        ])
        #penalty = theta_change * 0.04  # Penalize large theta changes
        #penalty += dist_to_goal * 2 # I add this, but it may not make sense
        #reward -= penalty  # Weight for smooth motion

    # Update state
        self.state = np.array([new_x, new_y, prev_state[0], prev_state[1], new_theta1, new_theta2, new_theta3])

        done = (dist_to_goal < 0.05) or (self.current_step >= self.max_steps)  # Stop when reaching goal

        return self.state, reward, done, {}


    def render(self, mode="human"):
        pass  # Visualization can be added if needed


class LSTMPPOPolicy(nn.Module):
    def __init__(self, input_size=7, hidden_size=256, output_size=2):  # Output is (delta_x, delta_y)
        super(LSTMPPOPolicy, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc_actor = nn.Linear(hidden_size, output_size)  # Action output (delta_x, delta_y)
        self.fc_critic = nn.Linear(hidden_size, 1)  # Value function for advantage estimation

    def forward(self, x, hidden):
        lstm_out, hidden = self.lstm(x, hidden)
        action_mean = self.fc_actor(lstm_out[:, -1, :])  # Output last time step
        value = self.fc_critic(lstm_out[:, -1, :])
        return action_mean, value, hidden

    def init_hidden(self):
        return (torch.zeros(1, 1, 256), torch.zeros(1, 1, 256))

def train_rl(model, env, num_episodes=1000, gamma=0.99, lr=0.000005, device="cuda"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    all_trajectories = []

    for episode in range(num_episodes):
        # Initialize hidden state at the start of each episode
        hidden_state = (
            torch.zeros(1, 1, 256).to(device),
            torch.zeros(1, 1, 256).to(device)
        )
        
        # Reset environment and get initial state
        state_np = env.reset()
        # Add batch and sequence dimensions: (1, 1, 7)
        state = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        trajectory = []
        # Append initial thetas from reset state
        initial_thetas = (state_np[4], state_np[5], state_np[6])
        trajectory.append(initial_thetas)
        
        log_probs, rewards, values = [], [], []
        done = False

        while not done:
            # Forward pass through LSTM
            action_mean, value, hidden_state = model(state, hidden_state)
            
            # Create normal distribution and sample action
            action_dist = distributions.Normal(action_mean, torch.tensor([0.1, 0.1], device=device))
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum(dim=-1)
            log_probs.append(log_prob)
            values.append(value)
            
            # Execute action (squeeze to handle batch and sequence dimensions)
            action_cpu = action.squeeze().detach().cpu().numpy()
            new_state_np, reward, done, _ = env.step(action_cpu)
            
            # Append new thetas to trajectory
            new_theta1, new_theta2, new_theta3 = new_state_np[4], new_state_np[5], new_state_np[6]
            trajectory.append((new_theta1, new_theta2, new_theta3))
            
            # Prepare next state with correct dimensions
            state = torch.tensor(new_state_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            rewards.append(reward)

        all_trajectories.append(trajectory)

        # Calculate discounted rewards
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(device)
        
        # Convert log_probs and values to tensors
        log_probs = torch.cat(log_probs)
        values = torch.cat(values).squeeze()
        
        # Compute advantages and losses
        advantages = discounted_rewards - values.detach()
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, discounted_rewards)
        loss = actor_loss + 0.2 * critic_loss
        
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 25 == 0:
            print(f"Episode {episode}, Total Reward: {sum(rewards)}, Loss: {loss.item()}")

    return all_trajectories

def generate_trajectory(
    lstm_policy,  # 训练好的LSTM策略网络
    predictor_model,  # 你的ResNetFCN预测模型
    start_pos,  # 初始位置 (x, y)
    start_thetas,  # 初始关节角度 (theta1, theta2, theta3)
    end_pos,  # 目标位置 (x, y)
    device,
    max_steps=100
):
    # 初始化环境
    env = RobotTrajectoryEnv(
        predictor_model,
        start_pos=start_pos,
        start_thetas=start_thetas,
        end_pos=end_pos,
        device=device,
        max_steps=max_steps
    )
    
    # 初始化LSTM隐藏状态
    hidden_state = (
        torch.zeros(1, 1, 128).to(device),
        torch.zeros(1, 1, 128).to(device)
    )
    
    state = env.reset()
    trajectory = [start_thetas]  # 初始角度
    
    # 转换为张量并添加batch和sequence维度
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    done = False
    step = 0
    
    with torch.no_grad():  # 禁用梯度计算
        while not done and step < max_steps:
            # 通过LSTM策略生成动作
            action_mean, _, hidden_state = lstm_policy(state_tensor, hidden_state)
            
            # 从正态分布采样动作（测试时可以直接使用均值）
            action = action_mean.squeeze().cpu().numpy()
            
            # 执行动作
            new_state, _, done, _ = env.step(action)
            
            # 记录新的关节角度
            new_thetas = (new_state[4], new_state[5], new_state[6])
            trajectory.append(new_thetas)
            
            # 更新状态张量
            state_tensor = torch.tensor(new_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            step += 1
    
    return trajectory
