
import numpy as np
def FK(lengths, angles):
    assert len(lengths) == len(angles), "Lengths and angles must have equal dimensions"
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    x = lengths.T @ cos_angles
    y = lengths.T @ sin_angles
    

    return x, y