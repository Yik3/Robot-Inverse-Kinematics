import matplotlib.pyplot as plt
import numpy as np

def visualize(lengths, angles):
    assert len(lengths) == len(angles), "Lengths and angles must have equal dimensions"

    plot_bounds = np.sum(lengths) + 1
    
    x_points = [0]
    y_points = [0]
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)


    for i in range(len(lengths)):
        x_points.append(x_points[-1] + lengths[i] * cos_angles[i])
        y_points.append(y_points[-1] + lengths[i] * sin_angles[i])

    plt.figure(figsize=(6, 6))
    plt.axis("equal")
    plt.xlim(-plot_bounds, plot_bounds)
    plt.ylim(-plot_bounds, plot_bounds)
    for i in range(len(x_points) - 1):
        segment_x = [x_points[i], x_points[i + 1]]
        segment_y = [y_points[i], y_points[i + 1]]
        plt.plot(segment_x, segment_y)

    plt.show()
