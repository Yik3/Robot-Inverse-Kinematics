import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from net_new import ResNetFCN  # your ResNetFCN definition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------
# Environment
# ------------------------------------------------
class RoboticArmEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, start_pos, start_angles, target_pos, max_steps=100):
        super().__init__()
        # Must match the 10 values returned by _get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=(2,), dtype=np.float32
        )
        
        self.start_pos = np.array(start_pos, dtype=np.float32)
        self.start_angles = np.array(start_angles, dtype=np.float32)
        self.target = np.array(target_pos, dtype=np.float32)
        self.max_steps = max_steps
        self.arm_lengths = [1.0, 1.0, 1.0]
        
        self.current_angles = None
        self.current_pos = None
        self.prev_angles = None
        self.steps = 0
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.current_angles = self.start_angles.copy()
        self.current_pos = self._fk(self.current_angles)
        self.prev_angles = self.current_angles.copy()
        
        return self._get_obs(), {}

    def _fk(self, angles):
        cumsum_angles = np.cumsum(angles)
        x = sum(l * np.cos(a) for l, a in zip(self.arm_lengths, cumsum_angles))
        y = sum(l * np.sin(a) for l, a in zip(self.arm_lengths, cumsum_angles))
        return np.array([x, y], dtype=np.float32)

    def step(self, action):
        action = np.clip(action, -0.1, 0.1)
        target_pos = self.current_pos + action
        
        # IK model
        with torch.no_grad():
            inputs = torch.FloatTensor([
                *target_pos,          # 2
                *self.current_pos,    # 2
                *self.prev_angles     # 3
            ]).to(device)              # total 7
            new_angles = model_pred(inputs.unsqueeze(0))[0].cpu().numpy()

        new_pos = self._fk(new_angles)
        angle_change = np.sum((self.prev_angles - new_angles) ** 2)
        reward = self._compute_reward(new_pos, angle_change)
        
        self.prev_angles = self.current_angles.copy()
        self.current_angles = new_angles
        self.current_pos = new_pos
        self.steps += 1
        
        terminated = np.linalg.norm(new_pos - self.target) < 0.05
        truncated = (self.steps >= self.max_steps) or (np.linalg.norm(new_pos) > 3.0)
        
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        # 2 + 3 + 2 + 3 = 10
        return np.concatenate([
            self.current_pos,
            self.current_angles,
            self.target - self.current_pos,
            self.prev_angles
        ]).astype(np.float32)

    def _compute_reward(self, new_pos, angle_change):
        dist = np.linalg.norm(new_pos - self.target)
        reward = -10.0 * dist - 0.5 * angle_change
        if dist < 0.05:
            reward += 100
        if np.linalg.norm(new_pos) > 3.0:
            reward -= 50
        return reward

# ------------------------------------------------
# Load your IK model
# ------------------------------------------------
model_pred = ResNetFCN(input_size=7, output_size=3).to(device)
model_pred.load_state_dict(torch.load("best_res_model.pth"))
model_pred.eval()

# ------------------------------------------------
# VecEnv creation
# ------------------------------------------------
start_position = [3, 0]
start_angles = [0, 0, 0]
target_position = [2.0, 1.0]

def make_env():
    return RoboticArmEnv(
        start_pos=start_position,
        start_angles=start_angles,
        target_pos=target_position,
        max_steps=200
    )

env = make_vec_env(make_env, n_envs=4)

# ------------------------------------------------
# Minimal LSTM Policy
# ------------------------------------------------
class LSTMPolicy(nn.Module):
    def __init__(self, input_dim=10, use_sde: bool = False,hidden_dim=128, action_dim=2):
        super().__init__()
        self.use_sde = use_sde
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

    # Hacky forward: just ignore if `obs` is a Box
    def forward(self, obs, hidden=None, episode_starts=None, deterministic=False, **kwargs):
        # 1) If SB3 is calling with a `Box`, just return dummy
        if isinstance(obs, gym.spaces.Box):
            # Return None or zero. It won't be used anyway.
            return None, None
        
        # 2) Convert to float tensor if not already
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        
        # 3) If obs is (batch_size, 10), add a time dimension => (batch, seq=1, 10)
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        
        # 4) Run LSTM
        x, new_hidden = self.lstm(obs, hidden)
        # 5) Actor head => scale by 0.1
        out = self.actor(x[:, -1, :]) * 0.1
        return out, new_hidden

policy = LSTMPolicy().to(device)

# ------------------------------------------------
# Train PPO with the policy instance
# ------------------------------------------------
model = PPO(
    policy=LSTMPolicy,   # passing the instantiated LSTMPolicy
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    verbose=1,
    device=device,
    use_sde=False   # crucial to avoid passing use_sde=True
)

model.learn(total_timesteps=500000)
model.save("arm_trajectory_planner")

# ------------------------------------------------
# Test
# ------------------------------------------------
model = PPO.load("arm_trajectory_planner", device=device)

test_env = RoboticArmEnv(
    start_pos=start_position,
    start_angles=start_angles,
    target_pos=target_position,
    max_steps=200
)

obs, _ = test_env.reset()
trajectory = {
    'angles': [test_env.current_angles.copy()],
    'positions': [test_env.current_pos.copy()],
    'rewards': []
}

hidden = (torch.zeros(1, 1, 128).to(device), 
          torch.zeros(1, 1, 128).to(device))

done = False
while not done:
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        action_tensor, hidden = model.policy(obs_tensor, hidden)
    action = action_tensor.cpu().numpy()[0]
    
    obs, reward, terminated, truncated, _ = test_env.step(action)
    trajectory['angles'].append(test_env.current_angles.copy())
    trajectory['positions'].append(test_env.current_pos.copy())
    trajectory['rewards'].append(reward)
    
    done = terminated or truncated

print(f"Steps: {len(trajectory['angles'])}")
print(f"Final position: {trajectory['positions'][-1]}")
print(f"Total reward: {sum(trajectory['rewards']):.2f}")

# Quick plot
def plot_trajectory(trajectory):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    angles = np.array(trajectory['angles'])
    axs[0].plot(angles[:, 0], label='Theta1')
    axs[0].plot(angles[:, 1], label='Theta2')
    axs[0].plot(angles[:, 2], label='Theta3')
    axs[0].set_ylabel('Joint Angles')
    axs[0].legend()

    positions = np.array(trajectory['positions'])
    axs[1].plot(positions[:, 0], positions[:, 1], label='Trajectory')
    axs[1].plot(start_position[0], start_position[1], 'go', label='Start')
    axs[1].plot(target_position[0], target_position[1], 'r*', label='Target')
    circle = plt.Circle((0, 0), 3, fill=False, linestyle='--')
    axs[1].add_artist(circle)
    axs[1].set_aspect('equal', 'box')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

plot_trajectory(trajectory)
