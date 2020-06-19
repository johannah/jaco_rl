
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
from IPython import embed
import numpy as np
import os
np.set_printoptions(precision=4, suppress=True)

f = 'results/test_start/test_start_random_jaco_00013_0000000512.pkl'
fo = open(f, 'rb')
replay = pickle.load(fo)

# control rate is .02
n = replay.num_steps_available()
states, actions, rewards, next_states, frames = replay.get_last_steps(n, return_frames=True)
out_path = f.replace('.pkl', '_kinematics_%04d.png')

joint_angles = states['joint_angles']
joint_extremes = states['joint_extremes'] 
print('plotting', n)
for t in range(n):
    plt.figure()
    plt.imshow(frames[t])
    title = """%d  
    DH  4:%s %.02f,%.02f,%.02f 7:%s %.02f,%.02f,%.02f
    """%(t,
                      int(np.rad2deg(joint_angles[t,4])), joint_extremes[t,0,0], joint_extremes[t,0,1], joint_extremes[t,0,2], 
                      int(np.rad2deg(joint_angles[t,6])), joint_extremes[t,2,0], joint_extremes[t,2,1], joint_extremes[t,2,2], 
                      )
    print(title)

    print(np.array(list(joint_angles[t].T)))
    print(np.array(list(actions[t])))
    plt.title(title)
    plt.savefig(out_path%t)
    plt.close()

frame_path = f.replace('.pkl', '_kinematics_*.png')
movie_path = frame_path[:-6]+'.mp4'
os.system("ffmpeg -pattern_type glob -i '%s' -c:v libx264 %s" %(frame_path, movie_path))

