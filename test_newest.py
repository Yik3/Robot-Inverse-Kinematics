import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from net_new import *

import gymnasium as gym
from gymnasium import spaces

class RoboticArmEnv(gym.Env):
    metadata = {'render_modes': ['human']}  # 可选：定义渲染模式
    
    def __init__(self, start_pos, start_angles, target_pos, max_steps=100):
        super().__init__()
        
        # 动作空间：末端执行器的坐标增量 (dx, dy)
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=(2,), dtype=np.float32
        )
        
        # 观测空间：当前状态 + 目标信息
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )
        
        # 初始化机械臂参数
        self.start_pos = np.array(start_pos)
        self.start_angles = np.array(start_angles)
        self.target = np.array(target_pos)
        self.max_steps = max_steps
        self.arm_lengths = [1.0, 1.0, 1.0]
        
        # 验证初始状态
        self.current_angles = None
        self.current_pos = None
        self.prev_angles = None
        self.steps = 0
        self.reset()

    def reset(self, seed=None, options=None):
        # 重置到初始状态
        self.current_angles = self.start_angles.copy()
        self.current_pos = self._fk(self.current_angles)
        self.prev_angles = self.current_angles.copy()
        self.steps = 0
        
        # 返回观测和info字典
        return self._get_obs(), {}

    def _fk(self, angles):
        cumulative_angles = np.cumsum(angles)
        x = sum(length * np.cos(angle) for length, angle in zip(self.arm_lengths, cumulative_angles))
        y = sum(length * np.sin(angle) for length, angle in zip(self.arm_lengths, cumulative_angles))
        return np.array([x, y])

    def step(self, action):
        # 处理动作
        target_delta = np.clip(action, -0.1, 0.1)
        target_pos = self.current_pos + target_delta
        
        # 调用逆运动学模型
        with torch.no_grad():
            inputs = torch.FloatTensor([
                *target_pos,
                *self.current_pos,
                *self.prev_angles
            ]).to(device)
            new_angles = model_pred(inputs.unsqueeze(0))[0].cpu().numpy()
        
        # 计算奖励
        new_pos = self._fk(new_angles)
        angle_change = np.sum((self.prev_angles - new_angles)**2)
        reward = self._compute_reward(new_pos, angle_change)
        
        # 更新状态
        self.prev_angles = self.current_angles.copy()
        self.current_angles = new_angles
        self.current_pos = new_pos
        self.steps += 1
        
        # 终止条件
        terminated = np.linalg.norm(new_pos - self.target) < 0.05
        truncated = (self.steps >= self.max_steps) or (np.linalg.norm(new_pos) > 3.0)
        
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        return np.concatenate([
            self.current_pos,
            self.current_angles,
            self.target - self.current_pos,
            self.prev_angles
        ]).astype(np.float32)

    def _compute_reward(self, new_pos, angle_change):
        distance = np.linalg.norm(new_pos - self.target)
        reward = -distance * 10  # 主奖励项
        reward -= 0.5 * angle_change  # 平滑性惩罚
        if distance < 0.05:
            reward += 100  # 成功奖励
        if np.linalg.norm(new_pos) > 3.0:  # 超出工作空间
            reward -= 50
        return reward

    def _check_done(self, pos):
        return (
            np.linalg.norm(pos - self.target) < 0.05 or
            np.linalg.norm(pos) > 3.0 or
            self.steps >= self.max_steps
        )
        
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# 设备设置 (确保与模型加载设备一致)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载逆运动学模型 (需提前定义ResNetFCN类)
model_pred = ResNetFCN(input_size=7, output_size=3).to(device)
model_pred.load_state_dict(torch.load("best_res_model.pth"))
model_pred.eval()

#-----------------------------------------------------------
# 强化学习环境配置
#-----------------------------------------------------------
# 用户定义的初始状态和目标
start_position = [3, 0]  # [x_start, y_start]
start_angles = [0, 0, 0]  # [theta1, theta2, theta3]
target_position = [2.0, 1.0]  # [x_end, y_end]

# 创建并行化环境 (需包裹环境创建函数)
def make_env():
    return RoboticArmEnv(
        start_pos=start_position,
        start_angles=start_angles,
        target_pos=target_position,
        max_steps=200
    )

# 创建4个并行环境
env = make_vec_env(make_env, n_envs=4)

#-----------------------------------------------------------
# LSTM策略网络定义 (匹配新观测空间)
#-----------------------------------------------------------
class LSTMPolicy(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=128, action_dim=2):  # 输入维度调整为11
        super().__init__()
        # 观测空间组成:
        # current_pos(2) + current_angles(3) + target_delta(2) + prev_angles(3) = 10
        # 实际输入为11维，发现计算错误时应检查_env.get_obs()的维度
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()  # 输出范围[-1,1]
        )
        
    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x, hidden)
        return self.actor(x[:, -1, :]) * 0.1, hidden  # 限制动作幅度并返回隐藏状态

#-----------------------------------------------------------
# 训练配置
#-----------------------------------------------------------
# 初始化策略
policy = LSTMPolicy().to(device)

# PPO参数配置
model = PPO(
    policy=policy,
    env=env,
    learning_rate=3e-4,
    n_steps=2048,        # 每个环境运行2048步后更新
    batch_size=64,       # 从经验池采样批量大小
    gamma=0.99,          # 折扣因子
    gae_lambda=0.95,     # GAE参数
    ent_coef=0.01,       # 熵系数
    verbose=1,
    device=device        # 确保与模型设备一致
)

# 训练模型
model.learn(total_timesteps=500000)  # 总步数=环境数×每个环境步数×epoch数

# 保存策略
model.save("arm_trajectory_planner")

#-----------------------------------------------------------
# 轨迹生成与验证
#-----------------------------------------------------------
# 加载训练好的模型 (需重新包装策略)
model = PPO.load("arm_trajectory_planer", device=device)

# 创建测试环境 (单环境)
test_env = RoboticArmEnv(
    start_pos=start_position,
    start_angles=start_angles,
    target_pos=target_position,
    max_steps=200
)

# 轨迹生成
obs = test_env.reset()
trajectory = {
    'angles': [test_env.current_angles.copy()],
    'positions': [test_env.current_pos.copy()],
    'rewards': []
}

# LSTM隐藏状态初始化
lstm_hidden = (torch.zeros(1, 1, 128).to(device), 
               torch.zeros(1, 1, 128).to(device))

while True:
    # 转换观测为PyTorch张量并添加时间步维度
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(device)
    
    # 使用LSTM策略需要传递隐藏状态
    with torch.no_grad():
        action, lstm_hidden = model.policy(obs_tensor, lstm_hidden)
    
    # 执行动作
    obs, reward, done, _ = test_env.step(action.cpu().numpy()[0])
    
    # 记录轨迹
    trajectory['angles'].append(test_env.current_angles.copy())
    trajectory['positions'].append(test_env.current_pos.copy())
    trajectory['rewards'].append(reward)
    
    if done:
        break

# 输出轨迹统计信息
print(f"轨迹步数: {len(trajectory['angles'])}")
print(f"最终位置: {trajectory['positions'][-1]}")
print(f"累计奖励: {sum(trajectory['rewards']):.2f}")

#-----------------------------------------------------------
# 轨迹可视化 (需要matplotlib)
#-----------------------------------------------------------
def plot_trajectory(trajectory):
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    # 绘制关节角度变化
    angles = np.array(trajectory['angles'])
    axs[0].plot(angles[:, 0], label='Theta1')
    axs[0].plot(angles[:, 1], label='Theta2')
    axs[0].plot(angles[:, 2], label='Theta3')
    axs[0].set_ylabel('Joint Angles (rad)')
    axs[0].legend()
    
    # 绘制末端轨迹
    positions = np.array(trajectory['positions'])
    axs[1].plot(positions[:, 0], positions[:, 1], 'b-', label='Trajectory')
    axs[1].plot(start_position[0], start_position[1], 'go', label='Start')
    axs[1].plot(target_position[0], target_position[1], 'r*', label='Target')
    
    # 绘制工作空间限制
    circle = plt.Circle((0, 0), 3, color='gray', fill=False, linestyle='--')
    axs[1].add_artist(circle)
    
    axs[1].set_xlim(-3.5, 3.5)
    axs[1].set_ylim(-3.5, 3.5)
    axs[1].set_aspect('equal')
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()

plot_trajectory(trajectory)