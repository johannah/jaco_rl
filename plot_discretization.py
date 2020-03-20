import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import pickle
from IPython import embed


def get_jaco_states(replay):
    """
     - reset moves all joint angle positions to 0 radians in dm_control
     - what does it do on the actual robot?
     - joint velocity approximates commanded velocity until some point is reached
     (probably max movement of joint) then velocity drops to near zero
     - after the first ~150 steps of random positive movements, it hits the end of
     movement
    """
    n_pos_bins = 100
    n_vel_bins = 40
    n_action_bins = 40
    pos_mat = np.arange(n_pos_bins*n_pos_bins, dtype=np.int32).reshape(n_pos_bins, n_pos_bins)
    arm_pos_states = replay.state[:max_ind,:n_joints*2]
    darm_pos_states = (((arm_pos_states+1)/2.0)*n_pos_bins).astype(np.int32)
    arm_vel_states = replay.state[:max_ind,26:26+n_joints]
    darm_vel_states = (((arm_vel_states+1)/2.0)*n_vel_bins).astype(np.int32)
    # build dataset in the form of:
    # sequence, bs, features
    actions = replay.action[:max_ind,:n_joints]
    dactions = (((actions+1)/2.0)*n_action_bins).astype(np.int32)
    ep_actions = dactions.ravel().reshape(n_episodes,ep_len,n_joints).swapaxes(0,1)
    ep_vel_states = darm_vel_states.ravel().reshape(n_episodes,ep_len,n_joints).swapaxes(0,1)
    ep_pos_states = darm_pos_states.ravel().reshape(n_episodes,ep_len,n_joints*2).swapaxes(0,1)
    for e in range(10):
        for an in range(n_joints):
            plt.figure()
            plt.plot(ep_actions[:,e,an], label='action')
            plt.plot(ep_vel_states[:,e,an], label='vel')
            plt.plot(ep_pos_states[:,e,an*2], label='pos1')
            plt.plot(ep_pos_states[:,e,(an*2)+1], label='pos2')
            plt.legend()
            plt.savefig('a%02de%d.png'%(an,e))
            plt.close()
if __name__ == '__main__':
    #replay = pickle.load(open('results/random_positive/random_jaco_00000_0000049999_buffer.pkl', 'rb'))
    replay = pickle.load(open('results/random_directed_reacher/random_directed_reacher_00000_0000049999_buffer.pkl', 'rb'))
    print('loaded replay buffer')
    ep_len = 500
    n_bins = 40
    n_joints = 2
    actions = (replay.action+1)/2.0
    # normalize bt 0 and 1 from -1 to 1
    d_actions = (actions*n_bins).astype(np.int32)
    max_ind = max([replay.ptr, replay.size])
    # make joints sequential
    n_episodes = max_ind//ep_len
    flat_ep_len = int(ep_len*n_joints)
    embed()


