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
from replay_buffer import ReplayBuffer, compress_frame
import plotting
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

def get_state_names_dict():
    plot_environment_kwargs = deepcopy(environment_kwargs)
    plot_environment_kwargs['flat_observation'] = False
    _plot_env = suite.load(domain_name=domain, task_name=task, task_kwargs=task_kwargs,  environment_kwargs=plot_environment_kwargs)
    plot_spec = _plot_env.observation_spec()
    del _plot_env; del plot_environment_kwargs
    state_names_dict = {}
    st = 0
    for key in plot_spec.keys():
        for ind in range(plot_spec[key].shape[0]):
            if len(plot_spec[key].shape) == 1:
                state_names_dict[key+'_%02d'%ind] = np.arange(st+ind, st+ind+1)
            else:
                state_names_dict[key+'_%02d'%ind] = np.arange(st+ind, st+ind+plot_spec[key].shape[1])
        st = st+ind+1
    return state_names_dict

def test_mujoco_controllers_absolute_step():
    print("starting evaluation for {} episodes".format(args.num_eval_episodes))
    # generate random seed

    eval_base_path = os.path.join(results_dir, '_absstep%s_tc%s'%(args.relative_step_size, args.eval_filename_modifier))
    eval_step_file_path = eval_base_path+'.cpkl' 
    eval_env = suite.load(domain_name=domain, task_name=task, task_kwargs=task_kwargs,  environment_kwargs=environment_kwargs)
    print("CREATING REPLAY eval", cam_dim)
    eval_replay_buffer = ReplayBuffer(kwargs['state_dim'], kwargs['action_dim'], 
                                 max_size=int(args.eval_replay_size), 
                                 cam_dim=cam_dim, seed=seed)
   
    error_dict = {}
    for jt in range(0,13):
        error_dict[jt] = 0
        done = False
        num_steps = 0
        reward = 0
        state_names_dict = get_state_names_dict()
        state_type, reward, discount, state = eval_env.reset()
        frame_compressed = get_next_frame(eval_env)
        obs_angles = deepcopy(state['observations'][3:13+3])
        total_errors = 0
        direction = 1
        obs_angles = deepcopy(state['observations'][3:13+3])
        action = deepcopy(obs_angles)
          
        num_steps = 0
        done = False
        last_change = num_steps
        while not done:
            steps_change =  num_steps - last_change

            if steps_change > 100:
                print('switch')
                direction = -direction
                last_change = num_steps
            elif obs_angles[jt]-(10*args.relative_step_size) <= amins[jt] and steps_change > 10:
                direction = 1
                print("direction", direction, action[jt], amins[jt], amaxes[jt])
                last_change = num_steps
            elif obs_angles[jt]+(10*args.relative_step_size) >= amaxes[jt] and steps_change > 10:
                direction = -1
                print("direction", direction, action[jt], amins[jt], amaxes[jt])
                last_change = num_steps
            action[jt] = obs_angles[jt] + args.relative_step_size*direction
            print('JT{}N{}A'.format(jt,num_steps),action)
            reward = 0
            # Perform action
            step_type, _, discount, next_state = eval_env.step(action)
            last_obs_angles = deepcopy(state['observations'][3:13+3])
            obs_angles = deepcopy(next_state['observations'][3:13+3])
            print('JT{}N{}O'.format(jt,num_steps),obs_angles)

            error = action-obs_angles
            abs_error = np.abs(error)
            if np.max(abs_error) > .1:
                print("----ERROR", error)
                error_dict[jt]+=1
                reward = -1

            next_frame_compressed = get_next_frame(eval_env)
            done = step_type.last()
            # Store data in replay buffer

            eval_replay_buffer.add(state['observations'], action, reward, 
                    next_state['observations'], done, 
                    frame_compressed=frame_compressed, 
                    next_frame_compressed=next_frame_compressed)

            frame_compressed = next_frame_compressed
            state = next_state
            num_steps +=1

        print("JT TOTAL ERRORS",error_dict)
        last_steps = eval_replay_buffer.get_last_steps(num_steps)
        done = True

        emovie_path = eval_base_path + '_JT%02d_%s_%s.mp4'%(jt, args.eval_filename_modifier, args.camera_view)
        ebase = emovie_path.replace('.mp4', '')

        plotting.plot_states(last_steps, ebase, detail_dict=state_names_dict)
        plotting.plot_position_actions(last_steps, ebase, relative=args.relative_step)
        plotting.plot_frames(emovie_path, eval_replay_buffer.get_last_steps(num_steps), plot_action_frames=True, min_action=-kwargs['max_action'], max_action=kwargs['max_action'], plot_frames=False)
        pickle.dump(eval_replay_buffer, open(ebase+'.epkl', 'wb'))
        plotting.plot_replay_reward(eval_replay_buffer, ebase, start_step=0, name_modifier='train')


    # write data files
    print("---------------------------------------")
    eval_replay_buffer.shrink_to_last_step()
    pickle.dump(eval_replay_buffer, open(eval_step_file_path, 'wb'))
    print("JT TOTAL ERRORS",error_dict)

    movie_path = eval_base_path + '_%s_%s.mp4'%(args.eval_filename_modifier,args.camera_view)
    plotting.plot_frames(movie_path, eval_replay_buffer.get_last_steps(eval_replay_buffer.size), plot_action_frames=True, min_action=-kwargs['max_action'], max_action=kwargs['max_action'], plot_frames=False)

def test_mujoco_controllers_relative_step():
    print("starting evaluation for {} episodes".format(args.num_eval_episodes))
    # generate random seed

    eval_base_path = os.path.join(results_dir, '_relstep%s_tc%s'%(args.relative_step, args.eval_filename_modifier))
    eval_step_file_path = eval_base_path+'.cpkl' 
    eval_env = suite.load(domain_name=domain, task_name=task, task_kwargs=task_kwargs,  environment_kwargs=environment_kwargs)
    print("CREATING REPLAY eval", cam_dim)
    eval_replay_buffer = ReplayBuffer(kwargs['state_dim'], kwargs['action_dim'], 
                                 max_size=int(args.eval_replay_size), 
                                 cam_dim=cam_dim, seed=seed)
    print(amins)
    print(amaxes)
    error_dict = {}
    for jt in range(0,13):
        error_dict[jt] = 0
        done = False
        num_steps = 0
        reward = 0
        state_names_dict = get_state_names_dict()
        state_type, reward, discount, state = eval_env.reset()
        frame_compressed = get_next_frame(eval_env)
        obs_angles = deepcopy(state['observations'][3:13+3])
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
        action = np.zeros(13)
        obs_angles = deepcopy(state['observations'][3:13+3])
        num_steps = 0
        done = False
        while not done:
            if obs_angles[jt]-(2*args.relative_step_size) <= amins[jt]:
                direction = 1
                print("direction", direction, action[jt], amins[jt], amaxes[jt])
            if obs_angles[jt]+(2*args.relative_step_size) >= amaxes[jt]:
                direction = -1
                print("direction", direction, action[jt], amins[jt], amaxes[jt])

            # turn hand so it is easier to see for finger moves
            #if num_steps < 10:
            #    action[1] = .1
            #    action[3] = .01
 
            #if jt == 4:
            #    if num_steps < 10:
            #        action[3] = .1
            #    else:
            #        action[3] = 0.0
 
            #if jt > 6:
            #    if num_steps < 10:
            #        action[5] = -.1
            #    else:
            #        action[5] = 0.0
            action[jt] = args.relative_step_size*direction
            #base_action[:7] = action
            print('JT{}N{}A'.format(jt,num_steps),action)
            reward = 0
            # Perform action
            step_type, _, discount, next_state = eval_env.step(action)
            last_obs_angles = deepcopy(state['observations'][3:13+3])
            obs_angles = deepcopy(next_state['observations'][3:13+3])
            print('JT{}N{}O'.format(jt,num_steps),obs_angles)

            error = (last_obs_angles+action)-obs_angles
            abs_error = np.abs(error)
            if np.max(abs_error) > .1:
                print("----ERROR", error)
                error_dict[jt]+=1
                reward = -1

            next_frame_compressed = get_next_frame(eval_env)
            done = step_type.last()
            # Store data in replay buffer

            eval_replay_buffer.add(state['observations'], action, reward, 
                    next_state['observations'], done, 
                    frame_compressed=frame_compressed, 
                    next_frame_compressed=next_frame_compressed)

            frame_compressed = next_frame_compressed
            state = next_state
            num_steps +=1

        print("JT TOTAL ERRORS",error_dict)
        last_steps = eval_replay_buffer.get_last_steps(num_steps)
        done = True

        emovie_path = eval_base_path + '_JT%02d_%s_%s.mp4'%(jt, args.eval_filename_modifier, args.camera_view)
        ebase = emovie_path.replace('.mp4', '')

        plotting.plot_states(last_steps, ebase, detail_dict=state_names_dict)
        plotting.plot_position_actions(last_steps, ebase, relative=args.relative_step)
        plotting.plot_frames(emovie_path, eval_replay_buffer.get_last_steps(num_steps), plot_action_frames=True, min_action=-kwargs['max_action'], max_action=kwargs['max_action'], plot_frames=False)
        pickle.dump(eval_replay_buffer, open(ebase+'.epkl', 'wb'))
        plotting.plot_replay_reward(eval_replay_buffer, ebase, start_step=0, name_modifier='train')


    # write data files
    print("---------------------------------------")
    eval_replay_buffer.shrink_to_last_step()
    pickle.dump(eval_replay_buffer, open(eval_step_file_path, 'wb'))
    print("JT TOTAL ERRORS",error_dict)

    movie_path = eval_base_path + '_%s_%s.mp4'%(args.eval_filename_modifier,args.camera_view)
    plotting.plot_frames(movie_path, eval_replay_buffer.get_last_steps(eval_replay_buffer.size), plot_action_frames=True, min_action=-kwargs['max_action'], max_action=kwargs['max_action'], plot_frames=False)

def get_kwargs(env):
    state_dim = env.observation_spec()['observations'].shape[0]
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
    parser.add_argument("--fixed_target_position", default=[-.8, .6, 1.2], type=list, help='specify target in some tasks')
    parser.add_argument("--eval_replay_size", default=int(10000), type=int, help='number of steps to store in replay buffer')
    parser.add_argument("-ne", "--num_eval_episodes", default=1, type=int, help='')
    parser.add_argument("--relative_step", default=False, action="store_true")
    parser.add_argument("--relative_step_size", default=.1, type=np.float)
    parser.add_argument("--episode_timelimit", default=3, type=np.float)
    parser.add_argument("--frame_height", default=400)
    parser.add_argument("--frame_width", default=480)
    parser.add_argument("--start_position", default='out')
    #parser.add_argument("--time_limit", default=10)                 # Time in seconds allowed to complete mujoco task
    parser.add_argument('-cv', '--camera_view', default=-1, help='camera view to use.') 
    parser.add_argument('-efm', '--eval_filename_modifier', default='_robot')
    parser.add_argument("--exp_name", default="pos_test")               
    parser.add_argument("--savedir", default="results", help='overall dir to store checkpoints')               

    args = parser.parse_args()
    results_dir = os.path.join(args.savedir, args.exp_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    environment_kwargs = {'flat_observation':True}
    domain = 'jaco'
    task = 'configurable_reacher'

    # need to figure out min/max of all joints
    joint_env = suite.load(domain_name=domain, task_name=task, 
                      task_kwargs={'relative_step':False,
                                   'degrees_of_freedom':13, 
                                   'start_position':args.start_position},  
                      environment_kwargs=environment_kwargs)

    amins = joint_env.action_spec().minimum
    amaxes = joint_env.action_spec().maximum

    task_kwargs = {}
    num_steps = 0
    file_name = "{}".format(args.exp_name)
    seed = args.seed
    task_kwargs = {
                   'relative_step':args.relative_step, 
                   'random':seed, 
                   'action_penalty':False,
                   'degrees_of_freedom':13,
                   'fixed_target_position':args.fixed_target_position,
                   'target_type':'fixed',
                   'physics_type':'mujoco', 
                    'episode_timelimit':args.episode_timelimit, 
                    'start_position':args.start_position}

    task_kwargs['fence'] = {'x':(-10,10), 'y':(-10, 10), 'z':(-10, 10)}


    _env = suite.load(domain_name=domain, task_name=task, task_kwargs=task_kwargs,  environment_kwargs=environment_kwargs)
    kwargs = get_kwargs(_env)
    del _env

    cam_dim = [args.frame_height, args.frame_width, 3]

    # Set seeds
    random_state = np.random.RandomState(seed)
    torch.manual_seed(seed)

    if args.relative_step:
        test_mujoco_controllers_relative_step()
    else:
        test_mujoco_controllers_absolute_step()





