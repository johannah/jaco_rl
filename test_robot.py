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
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment


def get_next_frame(frame_env):
    next_frame = None
    next_frame = frame_env.physics.render(height=args.frame_height,width=args.frame_width,camera_id=args.camera_view)
    next_frame = compress_frame(next_frame)
    return next_frame

def collect_mujoco_run():
    print("starting evaluation for {} episodes".format(args.num_eval_episodes))
    # generate random seed
    eval_step_file_path = load_model_path.replace('.pt', '%s.epkl'%args.eval_filename_modifier) 
    #if os.path.exists(eval_step_file_path):
    #    print('loading existing replay buffer:{}'.format(eval_step_file_path))
    #    eval_replay_buffer = load_replay_buffer(eval_step_file_path)
    #else:
    if 1:
        task_kwargs['physics_type'] = 'mujoco'
        eval_env = suite.load(domain_name=args.domain, task_name=args.task, task_kwargs=task_kwargs,  environment_kwargs=environment_kwargs)
        print("CREATING REPLAY eval", cam_dim)
        eval_replay_buffer = ReplayBuffer(kwargs['state_dim'], kwargs['action_dim'], 
                                     max_size=int(args.eval_replay_size), 
                                     cam_dim=cam_dim, seed=eval_seed)
        for e in range(args.num_eval_episodes):
            done = False
            num_steps = 0
            state_type, reward, discount, state = eval_env.reset()
            frame_compressed = get_next_frame(eval_env)
            angles = state['observations'][:7]
            rel_action = np.array([0., 0, 0, 0, 0, 0, 0])
            # TODO off by one error in step count!? of replay_buffer
            while done == False:
                #action = (
                #        policy.select_action(state['observations'])
                #    ).clip(-kwargs['max_action'], kwargs['max_action'])
                action = angles + rel_action
                action_deg = np.rad2deg(action)
                print('E{}N{}'.format(e,num_steps),action_deg)
                #action[6] = 0.1

                # Perform action
                step_type, reward, discount, next_state = eval_env.step(action)
                next_frame_compressed = get_next_frame(eval_env)
                done = step_type.last()
                # Store data in replay buffer

                if num_steps > 30:
                    done = True
                eval_replay_buffer.add(state['observations'], action, reward, 
                                  next_state['observations'], done, 
                                  frame_compressed=frame_compressed, 
                                  next_frame_compressed=next_frame_compressed)

                frame_compressed = next_frame_compressed
                state = next_state
                num_steps+=1
                time.sleep(.1)

            er = np.int(eval_replay_buffer.episode_rewards[-1])
            emovie_path = load_model_path.replace('.pt', '_eval_E{}_R{}_{}_{}.mp4'.format(e, er, args.eval_filename_modifier, args.camera_view))
            plot_frames(emovie_path, eval_replay_buffer.get_last_steps(num_steps), plot_action_frames=True, min_action=-kwargs['max_action'], max_action=kwargs['max_action'], plot_frames=True)
            pickle.dump(eval_replay_buffer, open(emovie_path.replace('.mp4', '.eepkl'), 'wb'))
        # write data files
        print("---------------------------------------")
        eval_replay_buffer.shrink_to_last_step()
        pickle.dump(eval_replay_buffer, open(eval_step_file_path, 'wb'))

    plot_replay_reward(eval_replay_buffer, load_model_path, start_step=train_step, name_modifier='eval')
    #movie_path = load_model_path.replace('.pt', '_eval_{}.mp4'.format(args.eval_filename_modifier))
    #plot_frames(movie_path, eval_replay_buffer.get_last_steps(eval_replay_buffer.size), plot_action_frames=True, min_action=-kwargs['max_action'], max_action=kwargs['max_action'], plot_frames=False)
    return eval_replay_buffer, eval_step_file_path

def replay_replay_buffer_on_robot(eval_replay_buffer, eval_step_file_path):
    print("starting robot rerun replay for {}".format(eval_step_file_path))
    # generate random seed
    random_state = np.random.RandomState(eval_seed)
    robot_step_file_path = eval_step_file_path.replace('.epkl', '.rpkl')
    if args.eval_filename_modifier not in robot_step_file_path:
        robot_step_file_path = eval_step_file_path.replace('.epkl', '%s.rpkl'%args.eval_filename_modifier)
    task_kwargs['physics_type'] = 'robot'
    robot_env = suite.load(domain_name=args.domain, task_name=args.task, task_kwargs=task_kwargs,  environment_kwargs=environment_kwargs)
    # hack
    robot_replay_buffer = ReplayBuffer(kwargs['state_dim'], kwargs['action_dim'], 
                                     max_size=int(args.eval_replay_size), 
                                     cam_dim=[0,0,0], seed=eval_seed)
    # TODO store target position somewhere!
#    robot_replay_buffer = ReplayBuffer(kwargs['state_dim'], kwargs['action_dim'], 
#                                     max_size=int(args.eval_replay_size), 
#                                     cam_dim=cam_dim, seed=eval_seed)
#
    eval_index = 0
    for e in range(eval_replay_buffer.episode_count):
        eval_done = False
        num_steps = 0
        state_type, reward, discount, state = robot_env.reset()
        # TODO off by one error in step count!? of replay_buffer
        while eval_done == False:
            # TODO hack assuming there is no wrap in replay buffer!
            # Perform action
            eval_action = eval_replay_buffer.actions[eval_index]
            eval_done = not eval_replay_buffer.not_dones[eval_index]
            print("EVAL ACTION", eval_action, num_steps)
            step_type, reward, discount, next_state = robot_env.step(eval_action)
            done = step_type.last()
            # Store data in replay buffer
            robot_replay_buffer.add(state['observations'], eval_action, reward, 
                              next_state['observations'], done, 
                              )

            state = next_state
            eval_index+=1
            num_steps+=1
            time.sleep(1)

    # write data files
    print("---------------------------------------")
    robot_replay_buffer.shrink_to_last_step()
    pickle.dump(robot_replay_buffer, open(robot_step_file_path, 'wb'))
    plot_replay_reward(robot_replay_buffer, load_model_path, start_step=train_step, name_modifier='robot')

    return robot_replay_buffer, robot_step_file_path

def load_replay_buffer(load_replay_path, load_model_path='', kwargs={}, seed=None):
    # load replay buffer

    if load_replay_path == 'empty' or (load_replay_path == '' and load_model_path == '' ):
        replay_buffer = ReplayBuffer(kwargs['state_dim'], kwargs['action_dim'], 
                                     max_size=int(args.replay_size), 
                                     cam_dim=cam_dim, seed=seed)
    else:
        if load_replay_path != '':
            if os.path.isdir(load_replay_path):
                print("searching for latest replay from dir: {}".format(load_replay_path))
                # find last file search_path = os.path.join(args.load_replay, '*.pkl') replay_files = glob(search_path)
                if not len(replay_files):
                    raise ValueError('could not find replay files at {}'.format(search_path))
                else:
                    load_replay_path = sorted(replay_files)[-1]
                    print('loading most recent replay from directory: {}'.format(load_replay_path))
            else:
                load_replay_path = load_replay_path
        else:
            load_replay_path = load_model_path.replace('.pt', '.pkl')
        print("loading replay from: {}".format(load_replay_path))
        replay_buffer = pickle.load(open(load_replay_path, 'rb'))
    return replay_buffer

def load_policy(load_model_path, kwargs={}):
    # Initialize policy
    start_step = 0
    if args.policy == "TD3":
        import TD3
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * kwargs['max_action']
        kwargs["noise_clip"] = args.noise_clip * kwargs['max_action']
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
    elif args.policy == 'bootstrap':
        from bdqn import BootstrapDDQN
        policy = BootstrapDDQN(**kwargs)
    elif args.policy == 'random':
        from utils import RandomPolicy
        policy = RandomPolicy(**kwargs)
    elif args.policy == "OurDDPG":
        import OurDDPG
        policy = OurDDPG.DDPG(**kwargs)
    elif args.policy == "DDPG":
        import DDPG
        policy = DDPG.DDPG(**kwargs)

    # create experiment directory (may not be used)
    exp_cnt = 0
    results_dir = os.path.join(args.savedir, args.exp_name+'%02d'%exp_cnt)
    while os.path.exists(results_dir):
        exp_cnt+=1
        results_dir = os.path.join(args.savedir, args.exp_name+'%02d'%exp_cnt)

    # load model if necessary
    if load_model_path != "":
        if os.path.isdir(load_model_path):
            print("loading latest model from dir: {}".format(load_model_path))
            # find last file
            search_path = os.path.join(load_model_path, '*.pt')
            model_files = glob(search_path)
            if not len(model_files):
                print('could not find model exp files at {}'.format(search_path))
                raise
            else:
                load_model_path = sorted(model_files)[-1]
        else:
            print("loading model from: {}".format(load_model_path))

        print('loading model from {}'.format(load_model_path))
        policy.load(load_model_path)
        try:
            start_step = int(load_model_path[-13:-3])
        except:
            try:
                start_step = policy.step
            except:
                print('unable to get start step from name - set it manually')

        # store in old dir
        if not args.continue_in_new_dir:
            results_dir = os.path.split(load_model_path)[0]
            print("continuing in loaded directory")
            print(results_dir)
        else:
            print("resuming in new directory")
            print(results_dir)
        #try:
        #    info = load_info_dict(load_model_path.replace('.pt', '.info'))
        #except:
        #    print('---not able to load info path')
        #    embed()

    else:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            print('storing results in: {}'.format(results_dir))
    return policy, start_step, load_model_path, results_dir

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
        "discount": args.discount,
        "tau": args.tau,
        "device":args.device,
    }
    return kwargs
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3", help='Policy name (TD3, DDPG or OurDDPG)')
    parser.add_argument("--domain", default="jaco", help='DeepMind Control Suite domain name')
    parser.add_argument("--task", default="reacher_easy", help='Deepmind Control Suite task name')
    parser.add_argument("--use_robot", default=True, action='store_true')
    parser.add_argument("--seed", default=0, type=int, help='random seed')
    parser.add_argument("--target_position", default=[-.2, .4, .3], type=list, help='specify target in some tasks')
    parser.add_argument("--save_freq", default=50000, type=int, help='how often to save model and replay buffer')
    parser.add_argument("--eval_replay_size", default=int(100000), type=int, help='number of steps to store in replay buffer')
    parser.add_argument("-ne", "--num_eval_episodes", default=1, type=int, help='')
    parser.add_argument("--max_timesteps", default=1e7, type=int, help='max time steps to run environment')
    parser.add_argument("--expl_noise", default=0.1, help='std of Gaussian exploration noise')
    parser.add_argument("--discount", default=0.99, help='discount factor')
    parser.add_argument("--batch_size", default=256, type=int, help='batch size for training agent')
    parser.add_argument("--frame_height", default=400)
    parser.add_argument("--frame_width", default=480)
    #parser.add_argument("--time_limit", default=10)                 # Time in seconds allowed to complete mujoco task
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int, help='freq of delayed policy updates')
    parser.add_argument('-cv', '--camera_view', default='topview', help='camera view to use.') 
    parser.add_argument('-efm', '--eval_filename_modifier', default='_robot')
    parser.add_argument('-d', '--device', default='cpu')   # use gpu rather than cpu for computation
    parser.add_argument('-fn', '--fence_name', default='jodesk', help='virtual fence name that prevents jaco from leaving this region.')   # use gpu rather than cpu for computation
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--load_replay", default="", help='Indicate replay buffer to load past experience from. Options are ["", "empty", file path of replay buffer.pkl]. If empty string, a new replay buffer will be created unless --load_model was invoked, in that case, the respective replay buffer will be loaded. If set to "empty", an new buffer will be created regardless of if --load_model was invoked.')
    parser.add_argument("--exp_name", default="test")               
    parser.add_argument("-ctn", "--continue_in_new_dir", default=False, action="store_true", help='If true, store results from loaded model in newly generated directory instead of resuming from loaded dir (possibly overwriting existing files).')               
    
    parser.add_argument("--savedir", default="results", help='overall dir to store checkpoints')               

    args = parser.parse_args()

    environment_kwargs = {'flat_observation':True}

    num_steps = 0
    file_name = "{}_{}_{}_{:05d}".format(args.exp_name, args.policy, args.domain, args.seed)
    print("---------------------------------------")
    print("Policy: {} Domain: {}, Task: {}, Seed: {}".format(args.policy, args.domain, args.task, args.seed))
    print("---------------------------------------")
    # info for particular task

    task_kwargs = {'xml_name':"jaco_j2s7s300_position.xml"}

    _env = suite.load(domain_name=args.domain, task_name=args.task, task_kwargs=task_kwargs,  environment_kwargs=environment_kwargs)
    kwargs = get_kwargs(_env)
    del _env

    # the real robot hits itself at 0.4141065692261898, 1.7588605071769132, 1.3986513249868833, 1.0891524583774879, 0.0390365195510298, 5.14543601786255, 0.6674719255901401, -0.0012319971190548208
    #bang_action = np.array([0.4141065692261898, 1.7588605071769132, 1.3986513249868833, 1.0891524583774879, 0.0390365195510298, 5.14543601786255, 0.6674719255901401, -0.0012319971190548208])
    if args.domain == 'jaco':
        if args.fence_name == 'jodesk':
            # .1f is too low - joint 4 hit sometimes!!!!
            task_kwargs['fence'] = {'x':(-.5,.5), 'y':(-1.0, .4), 'z':(.15, 1.2)}
        else:
            task_kwargs['fence'] = {'x':(-5,5), 'y':(-5, 5), 'z':(.15, 1.2)}

    cam_dim = [args.frame_height, args.frame_width, 3]

    # Set seeds
    policy, train_step, load_model_path, results_dir = load_policy(args.load_model, kwargs=kwargs)
    eval_seed = args.seed+train_step
    task_kwargs['random'] = eval_seed
    random_state = np.random.RandomState(eval_seed)
    torch.manual_seed(eval_seed)

    eval_replay_buffer, eval_step_file_path = collect_mujoco_run()
    replay_replay_buffer_on_robot(eval_replay_buffer, eval_step_file_path)





