import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import argparse
import os
import sys
import pickle
import utils
from replay_buffer import ReplayBuffer, compress_frame, plot_frames, plot_replay_reward
import time
from glob import glob

from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from dm_control import suite
from dm_control import viewer
from IPython import embed
from copy import deepcopy
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment


def get_next_frame(frame_env):
    next_frame = None
    next_frame = frame_env.physics.render(height=args.frame_height,width=args.frame_width,camera_id=args.camera_view)
    next_frame = compress_frame(next_frame)
    return next_frame

def test_mujoco_controllers():
    print("starting evaluation for {} episodes".format(args.num_eval_episodes))
    # generate random seed

    eval_base_path = os.path.join(results_dir, '_relstep%s_tc%s'%(args.relative_step, args.eval_filename_modifier))
    eval_step_file_path = eval_base_path+'.cpkl' 
    #if os.path.exists(eval_step_file_path):
    #    print('loading existing replay buffer:{}'.format(eval_step_file_path))
    #    eval_replay_buffer = load_replay_buffer(eval_step_file_path)
    #else:
    eval_env = suite.load(domain_name=domain, task_name=task, task_kwargs=task_kwargs,  environment_kwargs=environment_kwargs)
    print("CREATING REPLAY eval", cam_dim)
    eval_replay_buffer = ReplayBuffer(kwargs['state_dim'], kwargs['action_dim'], 
                                 max_size=int(args.eval_replay_size), 
                                 cam_dim=cam_dim, seed=seed)
    home_joint_angles =  [4.92,  # 283 deg
                         2.839, # 162.709854126
                         0,     # 0
                         .758,  # 43.43
                         4.6366,  # 265.66
                         4.493,   # 257.47
                         5.0249, # 287.9
                         10, 10, 10, 10, 10, 10]
    
    amins = eval_env.action_spec().minimum
    amaxes = eval_env.action_spec().maximum
    #for e in range(args.num_eval_episodes)
    print(amins)
    print(amaxes)
    error_dict = {}
    for jt in range(0,7):
        error_dict[jt] = 0
        done = False
        num_steps = 0
        reward = 0
        base_action = home_joint_angles
        state_type, reward, discount, state = eval_env.reset()
        frame_compressed = get_next_frame(eval_env)
        obs_angles = deepcopy(state['observations'][3:7+3])
        last_action = obs_angles
        total_errors = 0
        # WHAT I KNOW
        # if I dont step, nothing changes
        # when I run named.data.qpos[home] - the robot looks like i want
        # however if from that home, I "step" to "home" (the same angles) - the result is 45 offset
        """

        dm_control's rl/control.py var CONTROL_TIMESTEP is not ideal for position controllers.
        We need a timestep of .001 (specified in the xml file) for the physics solver in order to get it to work.
        At each timestep, the robot has CONTROL_TIMESTEP seconds to reach a particular position.  
        We interface with the real robot by calling an action client which blocks until the position is reached. 
        I think the best way to make these to processes appear the same is to carefully choose CONTROL_TIMESTEP and only allow
        sequential actions to be relatively near to each other so as to ensure that they can be completed within 
        CONTROL_TIMESTEP seconds in mujoco.


        The two systems will have different types of errors if mujoco is misconfigured.  

        The pararms that effect performance for position controllers (as far as I can tell) are: 
        CONTROL_TIMESTEP sent to rl/control.py (time allowed to reach goal)
        kv gain in the xml file
        physics timestep in xml file
        and how far the desired action is from the current position


        Here are some performance stats with those variables tested: 
        +1 error count if an observed position angle > .1 of the commanded angle


        """
        direction = 1
        for num_steps in range(200):
            if not done:
                # IT IS necessary to make a copy of observations because mujoco will use the reference 
                action = deepcopy(home_joint_angles[:7])
                #if obs_angles[jt] > 1.6:
                #if last_action[jt] >= amaxes[jt]-.5:
                #    action[jt] = last_action[jt]-relative_step
                #else:
                if last_action[jt] < .8*amins[jt]:
                    direction = 1
                    print("direction", direction, last_action[jt], amins[jt], amaxes[jt])
                if last_action[jt] > .8*amaxes[jt]:
                    direction = -1
                    print("direction", direction, last_action[jt], amins[jt], amaxes[jt])

                action[jt] = last_action[jt]+(args.relative_step*direction)

                base_action[:7] = action
                print('JT{}N{}A'.format(jt,num_steps),action)
                reward = 0
                # Perform action
                step_type, _, discount, next_state = eval_env.step(base_action)
                obs_angles = deepcopy(next_state['observations'][3:7+3])
                print('JT{}N{}O'.format(jt,num_steps),obs_angles)

                error = obs_angles-action
                abs_error = np.abs(error)
                if np.max(abs_error) > .1:
                    print("----ERROR", np.argmax(abs_error), error)
                    error_dict[jt]+=1
                    reward = -1

                last_action = action
                next_frame_compressed = get_next_frame(eval_env)
                done = step_type.last()
                # Store data in replay buffer

                eval_replay_buffer.add(state['observations'][:51], action[:7], reward, 
                        next_state['observations'][:51], done, 
                        frame_compressed=frame_compressed, 
                        next_frame_compressed=next_frame_compressed)

                frame_compressed = next_frame_compressed
                state = next_state

        print("JT TOTAL ERRORS",error_dict)
        done = True
        emovie_path = eval_base_path + '_JT{}_{}_{}.mp4'.format(jt, args.eval_filename_modifier, args.camera_view)
        plot_frames(emovie_path, eval_replay_buffer.get_last_steps(num_steps), plot_action_frames=True, min_action=-kwargs['max_action'], max_action=kwargs['max_action'], plot_frames=True)
        pickle.dump(eval_replay_buffer, open(emovie_path.replace('.mp4', '.eepkl'), 'wb'))
    # write data files
    print("---------------------------------------")
    eval_replay_buffer.shrink_to_last_step()
    pickle.dump(eval_replay_buffer, open(eval_step_file_path, 'wb'))
    print("JT TOTAL ERRORS",error_dict)
    return eval_replay_buffer, eval_step_file_path

def get_kwargs(env):
    #state_dim = env.observation_spec()['observations'].shape[0]
    state_dim = 51
    action_dim = env.action_spec().shape[0]
    min_action = float(env.action_spec().minimum[0])
    max_action = float(env.action_spec().maximum[0])
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
    }
    return kwargs
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int, help='random seed')
    parser.add_argument("--target_position", default=[-.8, .6, 1.2], type=list, help='specify target in some tasks')
    parser.add_argument("--eval_replay_size", default=int(10000), type=int, help='number of steps to store in replay buffer')
    parser.add_argument("-ne", "--num_eval_episodes", default=1, type=int, help='')
    parser.add_argument("--relative_step", default=.1, type=np.float)
    parser.add_argument("--frame_height", default=400)
    parser.add_argument("--frame_width", default=480)
    #parser.add_argument("--time_limit", default=10)                 # Time in seconds allowed to complete mujoco task
    parser.add_argument('-cv', '--camera_view', default='topview', help='camera view to use.') 
    parser.add_argument('-efm', '--eval_filename_modifier', default='_robot')
    parser.add_argument('-fn', '--fence_name', default='jodesk', help='virtual fence name that prevents jaco from leaving this region.')   # use gpu rather than cpu for computation
    parser.add_argument("--exp_name", default="pos_test")               
    parser.add_argument("--savedir", default="results", help='overall dir to store checkpoints')               

    args = parser.parse_args()
    results_dir = os.path.join(args.savedir, args.exp_name)
    environment_kwargs = {'flat_observation':True}
    domain = 'jaco'
    task = 'configurable_reacher'
    #task = 'relative_reacher_easy'

    num_steps = 0
    file_name = "{}".format(args.exp_name)
    seed = args.seed
    task_kwargs = {'xml_name':"jaco_j2s7s300_position.xml", 
                   'start_position':'home', 
                   'relative_step':False, 
                   'random':seed, 
                   'target_position':args.target_position,
                   'physics_type':'mujoco'}

    cam_dim = [args.frame_height, args.frame_width, 3]
    if args.fence_name == 'jodesk':
        # .1f is too low - joint 4 hit sometimes!!!!
        task_kwargs['fence'] = {'x':(-.5,.5), 'y':(-1.0, .4), 'z':(.15, 1.2)}
    else:
        task_kwargs['fence'] = {'x':(-5,5), 'y':(-5, 5), 'z':(.15, 1.2)}



    _env = suite.load(domain_name=domain, task_name=task, task_kwargs=task_kwargs,  environment_kwargs=environment_kwargs)
    kwargs = get_kwargs(_env)
    del _env

    am_dim = [args.frame_height, args.frame_width, 3]

    # Set seeds
    random_state = np.random.RandomState(seed)
    torch.manual_seed(seed)
    eval_replay_buffer, eval_step_file_path = test_mujoco_controllers()





