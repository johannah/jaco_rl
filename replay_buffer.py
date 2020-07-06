import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
np.set_printoptions(precision=3, suppress=True)
import time
import os
import zlib
import collections

from dm_control import suite
from skvideo.io import vwrite
from IPython import embed


def compress_frame(frame):
    return zlib.compress(frame.tostring())

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6), cam_dim=[0], seed=1939, garbage_check_multiplier=1.0):
        self.random_state = np.random.RandomState(seed)
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.episode_count = 0
        self.episodic_reward = 0
        self.episode_start_steps = [0]
        self.episode_start_times = [time.time()]
        self.episode_rewards = []
        self.states = np.zeros((self.max_size,state_dim), dtype=np.float32)
        self.next_states = np.zeros((self.max_size,state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), np.float32)
        self.rewards = np.zeros((max_size, 1), np.float32)
        self.not_dones = np.zeros((max_size, 1), dtype=np.int32)
        # always store next frames
        self.cam_dim = cam_dim
        self.frames = {}; self.next_frames = {}

        if cam_dim[0] > 0:
            self.frames_enabled = True
        else:
            self.frames_enabled = False
            # make fake frame batch
            self.fake_dim = list(cam_dim)
            self.fake_dim.insert(0, 32)
            self.fake_frames = np.zeros(self.fake_dim, np.uint8)
             
    def shrink_to_last_step(self):
        """ shrink size of replay to exactly fit data. useful when saving evaluation buffers"""
        self.states[:self.size]
        self.next_states[:self.size]
        self.actions[:self.size]
        self.rewards[:self.size]
        self.not_dones[:self.size]
        self.max_size = self.size

    def num_steps_available(self):
        if self.size < self.max_size:
            return self.ptr
        else:
            return self.max_size

    def undo_frame_compression(self, z):
        return np.frombuffer(zlib.decompress(z), dtype=np.uint8).reshape(self.cam_dim)

    def add(self, state, action, reward, next_state, done, frame_compressed=None, next_frame_compressed=None):
        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.not_dones[self.ptr] = 1. - done

        if self.frames_enabled:
            self.frames[self.ptr] = frame_compressed
            self.next_frames[self.ptr] = next_frame_compressed

        # track episode rollovers
        self.episodic_reward += reward
        if done:
            self.episode_count += 1
            self.episode_start_steps.append(self.size)
            self.episode_start_times.append(time.time())
            self.episode_rewards.append(self.episodic_reward)
            e_time = self.episode_start_times[-1]-self.episode_start_times[-2]
            e_steps = self.episode_start_steps[-1]-self.episode_start_steps[-2]
            print('EPISODE {} END: Ep R: {} Ep Time: {} Ep Steps:{} Total Steps: {}'.format(self.episode_count, 
                                                                                            self.episodic_reward, 
                                                                                            e_time, e_steps, self.size))
            self.episodic_reward = 0

        self.ptr = (self.ptr + 1) % self.max_size
        self.size+=1
   
    def get_last_steps(self, num_steps_back):
        assert num_steps_back>0
        if self.num_steps_available() < num_steps_back:
            return self.get_last_steps(self.num_steps_available())
         # can wrap around or dont need to wrap around
        indexes = np.arange(self.ptr-num_steps_back, self.ptr)
        return self.get_indexes(indexes)

    def get_indexes(self, batch_indexes):
        if self.frames_enabled:
            _frames = np.array([self.undo_frame_compression(self.frames[x]) for x in batch_indexes])
            _next_frames = np.array([self.undo_frame_compression(self.next_frames[x]) for x in batch_indexes])
            return self.states[batch_indexes], self.actions[batch_indexes], self.rewards[batch_indexes], self.next_states[batch_indexes], self.not_dones[batch_indexes], _frames, _next_frames
        else:
            if self.fake_frames.shape[0] != len(batch_indexes):
                self.fake_dim[0] = len(batch_indexes)
                self.fake_frames = np.zeros(self.fake_dim, np.uint8)
            return self.states[batch_indexes], self.actions[batch_indexes], self.rewards[batch_indexes], self.next_states[batch_indexes], self.not_dones[batch_indexes], self.fake_frames, self.fake_frames

    def sample(self, batch_size):
        indexes = self.random_state.randint(0,self.num_steps_available(),batch_size)
        return self.get_indexes(indexes)


        
def plot_frames(movie_fpath, last_steps, plot_pngs=False, plot_action_frames=True, min_action=-.8, max_action=.8, min_reward=-1, max_reward=1):
    st, ac, re, nst, nd, fr, nfr = last_steps
    n_steps = ac.shape[0]
    if plot_action_frames:
        n_actions = ac.shape[1]
        n_bars = n_actions + 1
        _, hsize, fr_wsize, nc = fr.shape
        wsize = int(fr_wsize*.40)
        n_bins = wsize
        cp = n_bins//2
        hw = hsize//n_bars
        action_bins = np.linspace(min_action, max_action, n_bins)
        reward_bins = np.linspace(min_reward, max_reward, n_bins)
        canvas = np.zeros((n_steps, hsize, wsize+1, nc), dtype=np.uint8)
        viridis = cm.get_cmap('viridis', n_actions)
        action_colors = np.array([np.array(viridis(x)[:nc]) for x in  np.linspace(0, 1, n_actions)])
        action_colors = (action_colors*255).astype(np.uint8)
        # red rewward
        reward_color = np.array([255,0,0])[:nc]
        try:
            for s in range(n_steps):
                for an,av in enumerate(ac[s]):
                    b = np.digitize(av, action_bins, right=True)
                    c = action_colors[an]
                    inds = np.linspace(cp-(cp-b), cp, np.abs(cp-b)+1, dtype=np.int)
                    canvas[s,hw*an:(hw*an)+hw,inds] = c
                rb = np.digitize(re[s][0], reward_bins, right=True)
                rinds = np.linspace(cp-(cp-rb), cp, np.abs(cp-rb)+1, dtype=np.int)
                canvas[s,hw*(n_bars-1):(hw*n_bars),rinds] = reward_color
            output = np.concatenate((fr, canvas, nfr), 2)
            vwrite(movie_fpath, output)
        except Exception as e:
            print(e)
            embed()
    else:
        vwrite(movie_fpath, fr)
    if plot_pngs:
        dir_path = movie_fpath.replace('.mp4', '')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        for n in range(fr.shape[0]):
            f,ax = plt.subplots(1,2)
            ax[0].imshow(fr[n])
            ax[1].imshow(nfr[n])
            action_title = ''
            for xn, a in enumerate(ac[n]):
                action_title += ' %.02f'%a
                if not xn%3:
                    action_title += '\n' 
            ax[0].set_title(action_title)
            ax[1].set_title("S:%s R:%s"%(n,re[n]))
            img_path = os.path.join(dir_path, 'frame_%05d.png'%n)
            print('writing', img_path)
            plt.savefig(img_path)
            plt.close()
        cmd = "ffmpeg -pattern_type glob -i '%s' -c:v libx264 '%s' -y"%(os.path.join(dir_path, '*.png'), os.path.join(dir_path, '_movie.mp4'))
        print('calling {}'.format(cmd))
        os.system(cmd)

def rolling_average(a, n=5) :
    if n == 0:
        return a
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_replay_reward(replay_buffer, load_model_path, start_step=0, name_modifier=''):

    st = np.array(replay_buffer.episode_start_times)
    plt.figure()
    plt.title("Episode Time")
    plt.plot(st[1:]-st[:-1])
    plt.savefig(load_model_path.replace('.pt', '_seconds_episode_%s.png'%name_modifier))
    plt.xlabel('episode')
    plt.ylabel('seconds')
    plt.close()

    plt.figure()
    plt.title("filtered reward")
    plt.plot(rolling_average(replay_buffer.episode_rewards, 10))
    plt.savefig(load_model_path.replace('.pt', '_rewards_episode_filt_%s.png'%name_modifier))
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.close()

    plt.figure()
    plt.title("reward")
    plt.plot(replay_buffer.episode_rewards)
    plt.savefig(load_model_path.replace('.pt', '_rewards_episode_%s.png'%name_modifier))
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.close()

    plt.figure()
    plt.title("cumulative reward")
    plt.plot(np.cumsum(replay_buffer.episode_rewards))
    plt.savefig(load_model_path.replace('.pt', '_cumulative_episode_%s.png'%name_modifier))
    plt.xlabel('episode')
    plt.ylabel('total reward')
    plt.close()

    plt.figure()
    plt.title("reward")
    plt.plot(np.array(replay_buffer.episode_start_steps[1:])+start_step, replay_buffer.episode_rewards)
    plt.savefig(load_model_path.replace('.pt', '_rewards_step_%s.png'%name_modifier))
    plt.xlabel('steps')
    plt.ylabel('reward')
    plt.close()

    plt.figure()
    plt.title("cumulative reward")
    plt.plot(np.array(replay_buffer.episode_start_steps[1:])+start_step, np.cumsum(replay_buffer.episode_rewards))
    plt.savefig(load_model_path.replace('.pt', '_cumulative_step_%s.png'%name_modifier))
    plt.xlabel('steps')
    plt.ylabel('total reward')
    plt.close()

# TODO fix frame
def test_fake_replay():
    # test replay
    state_dim = 2 
    action_dim = 2
    max_size = int(1e6)
    test_size = int(max_size*2)
    cam_dim = [0,0,0]
    # make basic state on, bc we will add index in
    basic_state = random_state.rand(test_size+1).astype(np.float32)
    basic_action = random_state.rand(test_size, action_dim).astype(np.float32)
    replay = ReplayBuffer(state_dim, action_dim, max_size, cam_dim, seed=1939)
    basic_rewards = []
    basic_dones = []

    def get_fake_r(step_num):
        if not step_num%2:
            return 0
        else:
            return 1

    def get_fake_done(step_num):
        if not step_num%5:
            return 0
        else:
            return 1

    for i in range(1, test_size):
        st = np.hstack((basic_state[i-1], [i-1]))
        nst = np.hstack((basic_state[i], [i]))
        replay.add(st, basic_action[i], get_fake_r(i), nst, get_fake_done(i))
        for _ in range(5):
            ss, sa, sr, sns, snd = replay.sample(1)
            sind = int(ss[0,1])
            nsind = int(sns[0,1])
            sval= ss[0,0]
            nsval = sns[0,0]
            assert sind + 1 == nsind
            assert sval == basic_state[sind]
            assert nsval == basic_state[nsind]
            assert sr == get_fake_r(nsind)
            assert not snd == get_fake_done(nsind)

 
if __name__ == '__main__':
    seed = 22
    max_size = int(1e7)
    test_size = int(max_size*2)
 
    random_state = np.random.RandomState(seed)
           
    frame_width = 120
    frame_height = 120
    cam_dim = [frame_height, frame_width, 3]

    env = suite.load(domain_name='reacher', task_name='easy', task_kwargs={'random':seed},
                                                              environment_kwargs={'flat_observation':True})
    state_dim = env.observation_spec()['observations'].shape[0]
    action_dim = env.action_spec().shape[0]
    replay = ReplayBuffer(state_dim, action_dim, max_size, cam_dim, seed=1939)
    min_action = float(env.action_spec().minimum[0])
    max_action = float(env.action_spec().maximum[0])
    state_type, reward, discount, state = env.reset()
    frame_compressed = replay.compress_frame(env.physics.render(height=frame_height,width=frame_width))
    for i in range(test_size):
        action = random_state.uniform(low=min_action, high=max_action, size=action_dim)
        step_type, reward, discount, next_state = env.step(action)
        done = step_type.last()
        next_frame_compressed = replay.compress_frame(env.physics.render(height=frame_height,width=frame_width))
        replay.add(state['observations'], action, reward, next_state['observations'], done, frame_compressed=frame_compressed, next_frame_compressed=next_frame_compressed)
        if done:
            state_type, reward, discount, state = env.reset()
            frame = replay.compress_frame(env.physics.render(height=frame_height,width=frame_width))
        else:
            next_state = state
            frame_compressed = next_frame_compressed
        if i > 10:
            replay.sample(4)




