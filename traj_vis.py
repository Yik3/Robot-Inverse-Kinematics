import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class ArmVisualizer:
    def __init__(self, arm_lengths, angles_sequence, angle_mode='absolute'):
        """
        参数：
        arm_lengths : tuple (l1, l2, l3) 三个连杆的长度
        angles_sequence : list of tuples [(theta1, theta2, theta3), ...]
        angle_mode : 'absolute' 表示所有角度基于水平面，'relative' 表示相对前一个关节
        """
        self.arm_lengths = arm_lengths
        self.l1, self.l2, self.l3 = arm_lengths
        self.angles = angles_sequence
        self.angle_mode = angle_mode
        self.history = []
        
        # 预计算所有坐标
        self.joint_positions = self._calculate_all_positions()

    def _calculate_positions(self, angles):
        """计算单个时间点的关节坐标（角度单位为度）"""
        theta1, theta2, theta3 = np.radians(angles)
        
        # 基座坐标
        x0, y0 = 0, 0
        
        # 第一段连杆（始终基于水平面）
        x1 = self.l1 * np.cos(theta1)
        y1 = self.l1 * np.sin(theta1)
        
        # 第二段连杆（根据模式选择角度基准）
        if self.angle_mode == 'absolute':
            # theta2基于水平面
            x2 = x1 + self.l2 * np.cos(theta2)
            y2 = y1 + self.l2 * np.sin(theta2)
        else:  # relative模式（原逻辑）
            theta_total_2 = theta1 + theta2
            x2 = x1 + self.l2 * np.cos(theta_total_2)
            y2 = y1 + self.l2 * np.sin(theta_total_2)
        
        # 第三段连杆（保持相对前一个关节）
        theta_total_3 = theta3  # 假设theta3相对第二个关节
        x3 = x2 + self.l3 * np.cos(theta_total_3)
        y3 = y2 + self.l3 * np.sin(theta_total_3)
        
        return (x0, y0), (x1, y1), (x2, y2), (x3, y3)

    def _calculate_all_positions(self):
        return [self._calculate_positions(a) for a in self.angles]

    def plot_configuration(self, frame_num, ax=None):
        """绘制指定帧的静态结构"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = self.joint_positions[frame_num]
        
        # 清除并绘制
        ax.clear()
        ax.plot([x0, x1], [y0, y1], 'b-', lw=4, label='Link 1')
        ax.plot([x1, x2], [y1, y2], 'g-', lw=4, label='Link 2')
        ax.plot([x2, x3], [y2, y3], 'r-', lw=4, label='Link 3')
        ax.scatter([x0, x1, x2, x3], [y0, y1, y2, y3], c='k', s=100, zorder=5)
        
        # 轨迹显示
        current_history = [pos[3] for pos in self.joint_positions[:frame_num+1]]
        if current_history:
            hist_x, hist_y = zip(*current_history)
            ax.plot(hist_x, hist_y, 'r--', alpha=0.5, label='End Effector Path')
        
        # 图形设置
        max_length = sum(self.arm_lengths)
        ax.set_xlim(-max_length*1.2, max_length*1.2)
        ax.set_ylim(-max_length*1.2, max_length*1.2)
        ax.set_aspect('equal')
        ax.set_title(f"Arm Configuration (Frame {frame_num})")
        ax.legend()
        return ax

    def create_animation(self, interval=100, save_path=None):
        """生成完整运动动画"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        def update(frame):
            (x0, y0), (x1, y1), (x2, y2), (x3, y3) = self.joint_positions[frame]
            
            ax.clear()
            # 绘制连杆
            ax.plot([x0, x1], [y0, y1], 'b-', lw=4)
            ax.plot([x1, x2], [y1, y2], 'g-', lw=4)
            ax.plot([x2, x3], [y2, y3], 'r-', lw=4)
            
            # 绘制关节
            ax.scatter([x0, x1, x2, x3], [y0, y1, y2, y3], c='k', s=100)
            
            # 绘制历史轨迹
            current_history = [pos[3] for pos in self.joint_positions[:frame+1]]
            if len(current_history) > 1:
                hx, hy = zip(*current_history)
                ax.plot(hx, hy, 'r--', alpha=0.5)
            
            # 坐标设置
            max_length = sum(self.arm_lengths)
            ax.set_xlim(-max_length*1.2, max_length*1.2)
            ax.set_ylim(-max_length*1.2, max_length*1.2)
            ax.set_aspect('equal')
            ax.set_title(f"Frame {frame}")

        ani = FuncAnimation(fig, update, frames=len(self.angles), 
                          interval=interval, blit=False)
        
        if save_path:
            ani.save(save_path, writer='pillow', fps=1000/interval)
            
        plt.show()
        return ani

# 使用示例 --------------------------------------------------
if __name__ == "__main__":
    # 测试数据（使用绝对角度模式）
    test_angles = [
        (0,0,0),
        (30, 45, -15),   # 第一帧
        (45, 60, -30),   # 第二帧
        (60, 75, -45),   # 第三帧
        (75, 90, -60)    # 第四帧
    ]
    
    # 初始化可视化工具（使用绝对角度模式）
    vis = ArmVisualizer(
        arm_lengths=(1, 1, 1),
        angles_sequence=test_angles,
        angle_mode='absolute'
    )
    
    # 绘制关键帧
    vis.plot_configuration(0)  # 第一帧
    plt.show()
    
    vis.plot_configuration(-1) # 最后一帧
    plt.show()
    
    # 生成动画（实时预览）
    vis.create_animation(interval=200, save_path="absolute_angle_arm.gif")