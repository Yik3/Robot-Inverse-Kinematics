import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class ArmVisualizer:
    def __init__(self, arm_lengths, angles_sequence, angle_mode='absolute', xy_sequence=None):

        self.arm_lengths = arm_lengths
        self.l1, self.l2, self.l3 = arm_lengths
        self.angles = angles_sequence
        self.angle_mode = angle_mode
        self.xy_sequence = xy_sequence 
        self.history = []
        

        self.joint_positions = self._calculate_all_positions()

    def _calculate_positions(self, angles):

        theta1, theta2, theta3 = np.radians(angles)
        

        x0, y0 = 0, 0
        

        x1 = self.l1 * np.cos(theta1)
        y1 = self.l1 * np.sin(theta1)
        
        if self.angle_mode == 'absolute':

            x2 = x1 + self.l2 * np.cos(theta2)
            y2 = y1 + self.l2 * np.sin(theta2)
        else:  
            theta_total_2 = theta1 + theta2
            x2 = x1 + self.l2 * np.cos(theta_total_2)
            y2 = y1 + self.l2 * np.sin(theta_total_2)
        
  
        theta_total_3 = theta3  
        x3 = x2 + self.l3 * np.cos(theta_total_3)
        y3 = y2 + self.l3 * np.sin(theta_total_3)
        
        return (x0, y0), (x1, y1), (x2, y2), (x3, y3)

    def _calculate_all_positions(self):
        return [self._calculate_positions(a) for a in self.angles]

    def plot_configuration(self, frame_num, ax=None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = self.joint_positions[frame_num]
        
        ax.clear()

        ax.plot([x0, x1], [y0, y1], 'b-', lw=4, label='Link 1')
        ax.plot([x1, x2], [y1, y2], 'g-', lw=4, label='Link 2')
        ax.plot([x2, x3], [y2, y3], 'r-', lw=4, label='Link 3')
        ax.scatter([x0, x1, x2, x3], [y0, y1, y2, y3], c='k', s=100, zorder=5)
        

        current_history = [pos[3] for pos in self.joint_positions[:frame_num+1]]
        if current_history:
            hist_x, hist_y = zip(*current_history)
            ax.plot(hist_x, hist_y, 'r--', alpha=0.5, label='End Effector Path (angles)')
        
        if self.xy_sequence is not None:

            xy_hist = self.xy_sequence[:frame_num+1]
            xy_x, xy_y = zip(*xy_hist)
            ax.plot(xy_x, xy_y, 'mo-', alpha=0.7, label='XY Trajectory')
        
        max_length = sum(self.arm_lengths)
        ax.set_xlim(-max_length*1.2, max_length*1.2)
        ax.set_ylim(-max_length*1.2, max_length*1.2)
        ax.set_aspect('equal')
        ax.set_title(f"Arm Configuration (Frame {frame_num})")
        ax.legend()
        return ax

    def create_animation(self, interval=100, save_path=None):

        fig, ax = plt.subplots(figsize=(8, 6))
        
        def update(frame):
            (x0, y0), (x1, y1), (x2, y2), (x3, y3) = self.joint_positions[frame]
            ax.clear()

            ax.plot([x0, x1], [y0, y1], 'b-', lw=4)
            ax.plot([x1, x2], [y1, y2], 'g-', lw=4)
            ax.plot([x2, x3], [y2, y3], 'r-', lw=4)

            ax.scatter([x0, x1, x2, x3], [y0, y1, y2, y3], c='k', s=100)
            

            current_history = [pos[3] for pos in self.joint_positions[:frame+1]]
            if len(current_history) > 1:
                hx, hy = zip(*current_history)
                ax.plot(hx, hy, 'r--', alpha=0.5)
            

            if self.xy_sequence is not None:
                xy_hist = self.xy_sequence[:frame+1]
                xy_x, xy_y = zip(*xy_hist)
                ax.plot(xy_x, xy_y, 'mo-', alpha=0.7)
            
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

# Example --------------------------------------------------
if __name__ == "__main__":
    test_angles = [
        (0, 0, 0),
        (30, 45, -15),
        (45, 60, -30),
        (60, 75, -45),
        (75, 90, -60)
    ]
    
    test_xy = [
        (3, 0),
        (2.9, 0.04),
        (3.0, 0.24),
        (3.0, 0.22),
        (2.97, 0.30)
    ]
    

    vis = ArmVisualizer(
        arm_lengths=(1, 1, 1),
        angles_sequence=test_angles,
        angle_mode='absolute',
        xy_sequence=test_xy
    )
    

    vis.plot_configuration(0)
    plt.show()
    
    vis.plot_configuration(len(test_angles)-1)
    plt.show()

    vis.create_animation(interval=200, save_path="absolute_angle_arm.gif")
