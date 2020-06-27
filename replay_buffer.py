import numpy as np
from IPython import embed

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6), cam_dim=[0,0,0], seed=1939):
        self.random_state = np.random.RandomState(seed)
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((self.max_size,state_dim), dtype=np.float32)
        self.next_states = np.zeros((self.max_size,state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), np.float32)
        self.rewards = np.zeros((max_size, 1), np.float32)
        self.not_dones = np.zeros((max_size, 1), dtype=np.bool)
        # always store next frames
        self.frames_enabled = False
        if cam_dim[0] > 0:
            self.frames = np.zeros((max_size, cam_dim[0], cam_dim[1], cam_dim[2]), dtype=np.uint8)

    def num_steps_available(self):
        # TODO - off by one? 
        if self.size < self.max_size:
            return self.ptr
        else:
            return self.max_size

    def add(self, state, action, next_state, reward, done, frame=None):
        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.not_dones[self.ptr] = 1. - done

        if self.frames_enabled:
            self.frames[self.ptr] = frame
        self.ptr = (self.ptr + 1) % self.max_size
        self.size+=1

    def get_last_steps(self, num_steps_back):
        print("calling num steps back", num_steps_back)
        assert num_steps_back>0
        if self.num_steps_available() < num_steps_back:
            return self.get_last_steps(self.num_steps_available())
         # can wrap around or dont need to wrap around
        indexes = np.arange(self.ptr-num_steps_back, self.ptr)
        return self.get_indexes(indexes, return_frames)

    def get_indexes(self, batch_indexes, return_frames=False):
        if return_frames:
            return self.states[batch_indexes], self.actions[batch_indexes], self.rewards[batch_indexes], self.next_states[batch_indexes], self.not_dones[batch_indexes], self.frames[batch_indexes]
        else:
            return self.states[batch_indexes], self.actions[batch_indexes], self.rewards[batch_indexes], self.next_states[batch_indexes],self.not_dones[batch_indexes],

    def sample(self, batch_size, return_frames=False):
        indexes = self.random_state.randint(0,self.ptr,batch_size)
        return self.get_indexes(indexes, return_frames)
        
# TODO fix frame


