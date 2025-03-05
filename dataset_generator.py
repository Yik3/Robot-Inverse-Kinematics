import numpy as np
import pandas as pd

# constants 
LINK_LENGTHS = np.array([1.0, 1.0, 1.0])

# how many evenly spaced discrete samples from [-pi, pi]
SAMPLES_PER_JOINT = 20 

# for each robot shape in the sweep, how many random nearby shapes are sampled?
NUM_NEARBY_SAMPLES = 20 

# a weight representing how far away the "nearby shapes" are
GAMMA = 0.1 # 

# perform forward kinematics given lengths and angles with respect to x axis
# lengths and angles are NP arrays
def FK(lengths, angles):
    assert len(lengths) == len(angles), "Lengths and angles must have equal dimensions"
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    x = lengths.T @ cos_angles
    y = lengths.T @ sin_angles
    

    return x, y

angle_sweep = np.linspace(-np.pi, np.pi, SAMPLES_PER_JOINT)

# this is some CRINGE for loop stuff. probably need vectorization if we need more arm segments

out = []
for t1 in angle_sweep:
    for t2 in angle_sweep:
        for t3 in angle_sweep:
            angles = np.array([t1, t2, t3])
            x, y = FK(LINK_LENGTHS, angles)
            for i in range(NUM_NEARBY_SAMPLES):
                # NOTE: these angles can exceed [-pi, pi]!!
                nearby_angles = [
                    t1 + GAMMA * np.random.randn(),
                    t2 + GAMMA * np.random.randn(),
                    t3 + GAMMA * np.random.randn()
                ]
                nearby_x, nearby_y = FK(LINK_LENGTHS, nearby_angles)
                datapoint = np.array([x, y, nearby_x, nearby_y, nearby_angles[0], nearby_angles[1], nearby_angles[2], t1, t2, t3])
                out.append(datapoint)

out_arr = np.vstack(out)
df = pd.DataFrame(out_arr, columns=["x", "y", "prev_x", "prev_y", "prev_theta1", "prev_theta2", "prev_theta3", "theta1", "theta2", "theta3"])
df.to_csv("3_segment_IK_dataset", index=False)


