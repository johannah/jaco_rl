import numpy as np
from IPython import embed

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6), cam_dim=[0,0,0]):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.states = {}; self.next_states = {}
        for state_key, state_details in state_dim.items():
            if len(state_details.shape) == 1:
                shape =[self.max_size] + list(state_details.shape) + [1]
            else:
                shape = [self.max_size] + list(state_details.shape)
            self.states[state_key] = np.empty((shape), dtype=state_details.dtype)
            self.next_states[state_key] = np.empty((shape), dtype=state_details.dtype)
        self.actions = np.zeros((max_size, action_dim))
        self.rewards = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        if cam_dim[0] > 0:
            self.frames = np.zeros((max_size, cam_dim[0], cam_dim[1], cam_dim[2]), dtype=np.uint8)

    def num_steps_available(self):
        # TODO - off by one? 
        if self.size < self.max_size:
            return self.ptr
        else:
            return self.max_size

    def add(self, state, action, next_state, reward, done, frame=None):
        for state_key in state.keys():
            if len(state[state_key].shape) == 1:
                self.states[state_key][self.ptr] = state[state_key][:,None]
                self.next_states[state_key][self.ptr] = next_state[state_key][:,None]
            else:
                self.states[state_key][self.ptr] = state[state_key]
                self.next_states[state_key][self.ptr] = next_state[state_key]

        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        if frame is not None:
            self.frames[self.ptr] = frame
        self.ptr = (self.ptr + 1) % self.max_size
        self.size+=1

    def get_last_steps(self, num_steps_back, return_frames=False):
        print("calling num steps back", num_steps_back)
        assert num_steps_back>0
        if self.num_steps_available() < num_steps_back:
            print("num steps available < num steps back",  self.num_steps_available(), num_steps_back)
            return self.get_last_steps(self.num_steps_available())
         # can wrap around or dont need to wrap around
        print('making inds', num_steps_back-self.ptr, self.ptr)
        ind = np.arange(self.ptr-num_steps_back, self.ptr)
        return self.get_sample(ind, return_frames)

    def get_sample(self, batch_indexes, return_frames=False):
        _states = {}
        _next_states = {}
        for state_key in self.states.keys():
            _states[state_key] = self.states[state_key][batch_indexes]
            _next_states[state_key] = self.next_states[state_key][batch_indexes]
        if return_frames:
            return _states, self.actions[batch_indexes], self.rewards[batch_indexes], _next_states, self.frames[batch_indexes]
        else:
            return _states, self.actions[batch_indexes], self.rewards[batch_indexes], _next_states
