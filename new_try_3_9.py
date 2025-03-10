import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from net_new import *

class RoboticArmEnv:
    def __init__(self, target_pos, max_steps=100):
        self.target = np.array(target_pos)
        self.max_steps = max_steps
        self.arm_lengths = [1.0, 1.0, 1.0]  # 三节臂长
        self.reset()

    def reset(self):
        # 初始状态：随机起点或固定起点
        self.current_angles = np.random.uniform(0, 2*np.pi, 3)  # theta1,2,3
        self.current_pos = self.forward_kinematics(self.current_angles)
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        # 观测：当前末端位置、关节角度、目标位置
        return np.concatenate([
            self.current_pos,
            self.current_angles,
            self.target
        ])

    def forward_kinematics(self, angles):
        # 计算正运动学（验证用）
        x = sum([self.arm_lengths[i] * np.cos(sum(angles[:i+1])) for i in range(3)])
        y = sum([self.arm_lengths[i] * np.sin(sum(angles[:i+1])) for i in range(3)])
        return np.array([x, y])

    def step(self, action):
        # 动作空间：目标末端位置的增量 (dx, dy)
        target_delta = np.clip(action, [-0.1], [0.1])  # 限制单步最大变化
        target_pos = self.current_pos + target_delta

        # 使用完美逆运动学模型预测关节角度
        with torch.no_grad():
            inputs = torch.FloatTensor([
                *target_pos,
                *self.current_pos,
                *self.current_angles
            ]).to(device)
            new_angles = model_pred(inputs.unsqueeze(0))[0].cpu().numpy()

        # 更新状态
        self.current_angles = new_angles
        new_pos = self.forward_kinematics(new_angles)
        self.current_pos = new_pos
        self.steps += 1

        # 计算奖励
        reward = self._compute_reward(new_pos)
        done = self._check_done(new_pos)
        return self._get_obs(), reward, done, {}

    def _compute_reward(self, new_pos):
        # 主要奖励：向目标靠近
        distance = np.linalg.norm(new_pos - self.target)
        reward = -distance  # 负距离作为奖励
        
        # 附加惩罚：关节角度突变
        angle_change = np.sum((self.current_angles - new_angles)**2)
        reward -= 0.1 * angle_change
        
        # 成功奖励
        if distance < 0.05:
            reward += 10
        return reward

    def _check_done(self, pos):
        # 终止条件：到达目标、超出工作空间、步数耗尽
        return (
            np.linalg.norm(pos - self.target) < 0.05 or
            np.linalg.norm(pos) > 3.0 or
            self.steps >= self.max_steps
        )
        
class LSTMPolicy(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, action_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()  # 输出范围[-1,1]
        )
        
    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x, hidden)
        action = self.actor(x[:, -1, :])  # 取最后时间步
        return action * 0.1  # 限制动作幅度
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_pred = ResNetFCN(input_size=7, output_size=3).to(device)
model_pred.load_state_dict(torch.load("best_res_model.pth"))
model_pred.eval()

# 创建环境
env = make_vec_env(lambda: RoboticArmEnv(target_pos=[2.0, 1.0]), n_envs=4)

# 初始化策略
policy = LSTMPolicy().to(device)
model = PPO(
    policy=policy,
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    verbose=1
)

# 训练
model.learn(total_timesteps=100000)

# 保存策略
model.save("arm_planner")

# 加载训练好的模型
model = PPO.load("arm_planner")

# 初始化环境
env = RoboticArmEnv(target_pos=[2.0, 1.0])
obs = env.reset()
trajectory = []

# 生成轨迹
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    trajectory.append(env.current_angles.copy())
    if done:
        break

# 输出结果
print(f"Generated trajectory with {len(trajectory)} steps:")
print(np.array(trajectory))