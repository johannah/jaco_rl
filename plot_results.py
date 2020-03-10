import pickle
from IPython import embed
import matplotlib
import numpy as np
from skvideo.io import vwrite, FFmpegWriter

f = 'results/imitate_robot_jaco_00000_0000001159.pkl'
fo = open(f, 'rb')
replay = pickle.load(fo)


# control rate is .02
num_frames = replay.ptr-1
exp = replay.get_last_experience(num_steps_back=num_frames)
actions = exp[1]
frames = np.array(exp[-1])
obs_shape = frames.shape[1:]
out_path = 'movie_%s.mp4'%num_frames
#vwrite(out_path, frames)

vid_out = FFmpegWriter('fast_'+out_path,
                       inputdict={'-r':'50'},
                       outputdict={'-r':'60',
                                   '-vcodec':'libx264'})
for idx in range(len(frames)):
    vid_out.writeFrame(frames[idx])
    #vid_out.writeFrame(np.ones((obs_shape), dtype=np.uint8))
vid_out.close()

