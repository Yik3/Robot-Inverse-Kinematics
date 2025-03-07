import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gym import spaces
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
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)

        # Observation space includes theta1, theta2, theta3 from the start
        self.observation_space = spaces.Box(
            low=np.array([-3, -3, -3, -3, -np.pi, -np.pi, -np.pi]),  # Min values
            high=np.array([3, 3, 3, 3, np.pi, np.pi, np.pi]),        # Max values
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

    # Update x, y with small changes
        new_x = np.clip(prev_state[0] + action[0][0], -3, 3)
        new_y = np.clip(prev_state[1] + action[0][1], -3, 3)

    # Move input tensor to the same device as predictor_model
        input_tensor = torch.tensor([new_x, new_y, prev_state[0], prev_state[1], prev_state[4], prev_state[5], prev_state[6]], dtype=torch.float32).unsqueeze(0).to(self.device)
    
    # Ensure predictor model is on CUDA
        new_theta1, new_theta2, new_theta3 = self.predictor_model(input_tensor).detach().cpu().numpy().squeeze()

    # Compute reward
        dist_to_goal = np.linalg.norm([new_x, new_y] - self.end_pos)
        prev_dist = np.linalg.norm([prev_state[0], prev_state[1]] - self.end_pos)

        reward = (prev_dist - dist_to_goal) * 1.5  # Reward for getting closer

    # Penalize large joint angle changes (minimizing θ changes for smoother motion)
        theta_change = np.linalg.norm([
            new_theta1 - prev_state[4],
            new_theta2 - prev_state[5],
            new_theta3 - prev_state[6]
        ])
        penalty = theta_change * 0.1  # Penalize large theta changes
        reward -= penalty  # Weight for smooth motion

    # Update state
        self.state = np.array([new_x, new_y, prev_state[0], prev_state[1], new_theta1, new_theta2, new_theta3])

        done = (dist_to_goal < 0.01) or (self.current_step >= self.max_steps)  # Stop when reaching goal

        return self.state, reward, done, {}


    def render(self, mode="human"):
        pass  # Visualization can be added if needed


class LSTMPPOPolicy(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, output_size=2):  # Output is (delta_x, delta_y)
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
        return (torch.zeros(1, 1, 128), torch.zeros(1, 1, 128))

def train_rl(model, env, num_episodes=1000, gamma=0.99, lr=0.0003, device="cuda"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)  # Move model to CUDA
    hidden_state = (torch.zeros(1, 1, 128).to(device), torch.zeros(1, 1, 128).to(device))  # Ensure hidden state is on GPU

    all_trajectories = []  # Store all trajectories

    for episode in range(num_episodes):
        state = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0).to(device)  # Move state to GPU
        log_probs, rewards, values = [], [], []
        trajectory = []  # Store (theta1, theta2, theta3) per step
        done = False

        while not done:
            '''
            if state.dim() == 2:
                state = state.unsqueeze(0)  # Ensure batch size for LSTM
            '''
            action_mean, value, hidden_state = model(state, hidden_state)
            action_dist = distributions.Normal(action_mean, torch.tensor([0.1, 0.1]).to(device))  # Move to GPU
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum()

            new_state, reward, done, _ = env.step(action.detach().cpu().numpy())  # Move action to CPU before passing to env

            # Store log_prob with batch dimension (Fixes concatenation issue)
            log_probs.append(log_prob.unsqueeze(0))  # Ensures shape (1,)

            values.append(value)  # Ensure values is a list of tensors
            rewards.append(reward)

            # Extract theta1, theta2, theta3 and store in trajectory
            theta1, theta2, theta3 = new_state[4], new_state[5], new_state[6]
            trajectory.append((theta1, theta2, theta3))

            state = torch.tensor(new_state, dtype=torch.float32).unsqueeze(0).to(device)  # Move to GPU

        all_trajectories.append(trajectory)  # Store full trajectory

        # Convert rewards to a tensor and ensure correct dtype
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(device)  # Convert to float32

        # Convert log_probs and values to tensors correctly
        log_probs = torch.cat(log_probs)  # No more dimension mismatch
        values = torch.cat(values).view(-1, 1)  # Ensure values has correct shape

        # FIX: Compute advantage properly without breaking the graph
        advantage = discounted_rewards - values.detach()  # FIXED  # Detach to avoid second backward pass

        # Loss function
        actor_loss = -(log_probs * advantage).mean()
        critic_loss = F.mse_loss(values, discounted_rewards.view(-1, 1))  # Ensure matching shapes
        loss = actor_loss + 0.5 * critic_loss  # Weighted sum

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 50 == 0:
            print(f"Episode {episode}, Reward: {sum(rewards)}")

    return all_trajectories  # Return all joint angle sequences over episodes