import numpy as np
import matplotlib.pyplot as plt

data = np.load('../test_dataset/2020-03-10-15-24-35_joint_state.npz')
plt.plot(data['joint_velocities'])
plt.show()

