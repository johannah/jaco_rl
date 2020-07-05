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

def equal_quantization(n_bins, actions):
     integers = (((actions+1.0)/2.0)*n_bins).astype(np.int32)
     return ((integers.astype(np.float32)/n_bins)*2)-1

def get_next_frame(frame_env):
    next_frame = None
    if args.state_pixels:
        if args.camera_view == '':
            next_frame = frame_env.physics.render(height=args.frame_height,width=args.frame_width)
        else:
            next_frame = frame_env.physics.render(height=args.frame_height,width=args.frame_width,camera_id=args.camera_view)
        if args.convert_to_gray:
            next_frame = img_as_ubyte(rgb2gray(next_frame)[:,:,None])
        next_frame = compress_frame(next_frame)
    return next_frame

def plot_loss_dict():
    loss_plot_path = load_model_path.replace('.pt', '_loss.png') 
    loss_dict = policy.get_loss_plot_data()
    plt.figure()
    for key, val in loss_dict.items():
        plt.plot(val[0], val[1], label=key)
    plt.title('Training Loss')
    plt.legend(loc=2)
    plt.savefig(loss_plot_path)
    plt.close()
    
def evaluate():
    print("starting evaluation for {} episodes".format(args.num_eval_episodes))

    plot_loss_dict()
    train_replay_path = load_model_path.replace('.pt', '.pkl')
    # TODO this isn't right - need to track steps in replay really

    train_replay_buffer = load_replay_buffer(train_replay_path)
    plot_replay_reward(train_replay_buffer, load_model_path, start_step=train_step, name_modifier='train')
 
    eval_seed = args.seed+10
    random_state = np.random.RandomState(eval_seed)
    eval_step_file_path = load_model_path.replace('.pt', '.epkl') 
    if os.path.exists(eval_step_file_path):
        eval_replay_buffer = load_replay_buffer(eval_step_file_path)
    else:
        eval_replay_buffer = ReplayBuffer(state_dim, action_dim, 
                                     max_size=int(args.eval_replay_size), 
                                     cam_dim=cam_dim, seed=eval_seed)
 
        eval_env = suite.load(domain_name=args.domain, task_name=args.task, task_kwargs={'random':eval_seed},  environment_kwargs=environment_kwargs)

        for e in range(args.num_eval_episodes):
            done = False
            num_steps = 0
            state_type, reward, discount, state = eval_env.reset()
            frame_compressed = get_next_frame(eval_env)
            # TODO off by one error in step count!? of replay_buffer
            while done == False:
                #action = (
                #        policy.select_action(state['observations'])
                #        + random_state.normal(0, max_action * args.expl_noise, size=action_dim)
                #    ).clip(-max_action, max_action)
 
                action = (
                        policy.select_action(state['observations'])
                    ).clip(-max_action, max_action)

                # Perform action
                step_type, reward, discount, next_state = eval_env.step(action)
                next_frame_compressed = get_next_frame(eval_env)
                done = step_type.last()
                # Store data in replay buffer
                eval_replay_buffer.add(state['observations'], action, reward, 
                                  next_state['observations'], done, 
                                  frame_compressed=frame_compressed, 
                                  next_frame_compressed=next_frame_compressed)

                frame_compressed = next_frame_compressed
                state = next_state
                num_steps+=1
                time.sleep(.1)
            movie_path = load_model_path.replace('.pt', '_eval%02d.mp4'%e) 
            plot_frames(movie_path, eval_replay_buffer.get_last_steps(num_steps), plot_action_frames=True, min_action=min_action, max_action=max_action)
            #plot_frames(movie_path, eval_replay_buffer.get_last_steps(num_steps), plot_pngs=True, plot_action_frames=True, min_action=min_action, max_action=max_action)
 
   #     write data files
        print("---------------------------------------")
        eval_replay_buffer.shrink_to_last_step()
        pickle.dump(eval_replay_buffer, open(eval_step_file_path, 'wb'))

    plot_replay_reward(eval_replay_buffer, load_model_path, start_step=train_step, name_modifier='eval')
    return eval_replay_buffer, eval_step_file_path

def load_replay_buffer(load_replay_path, load_model_path=''):
    # load replay buffer

    if load_replay_path == 'empty' or (load_replay_path == '' and load_model_path == '' ):
        replay_buffer = ReplayBuffer(state_dim, action_dim, 
                                     max_size=int(args.replay_size), 
                                     cam_dim=cam_dim, seed=args.seed)
    else:
        if load_replay_path != '':
            if os.path.isdir(load_replay_path):
                print("searching for latest replay from dir: {}".format(load_replay_path))
                # find last file
                search_path = os.path.join(args.load_replay, '*.pkl')
                replay_files = glob(search_path)
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

def train(start_t):
    random_state = np.random.RandomState(args.seed)
    replay_buffer = load_replay_buffer(load_replay_path=args.load_replay, load_model_path=load_model_path)
    info = utils.create_new_info_dict(args, load_model_path, args.load_replay)
    done = False
    state_type, reward, discount, state = env.reset()
    frame_compressed = get_next_frame(env)
    for t in range(start_t, int(args.max_timesteps)):
        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = random_state.uniform(low=min_action, high=max_action, size=action_shape)
        else:
            # Select action randomly or according to policy
            action = (
                    policy.select_action(state['observations'])
                    + random_state.normal(0, max_action * args.expl_noise, size=action_dim)
                ).clip(-max_action, max_action)
        # Perform action
        step_type, reward, discount, next_state = env.step(action)
        next_frame_compressed = get_next_frame(env)
        done = step_type.last()
        # Store data in replay buffer
        replay_buffer.add(state['observations'], action, reward, next_state['observations'], done, frame_compressed=frame_compressed, next_frame_compressed=next_frame_compressed)
        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(t, replay_buffer, args.batch_size)
        # prepare for next step
        if not done:
            state = next_state
            frame_compressed = next_frame_compressed
        else:
            print("Total T: {} Episode Num: {} Reward: {}".format(t, replay_buffer.episode_count-1, replay_buffer.episode_rewards[-1]))
            print("---------------------------------------")
            # Reset environment
            state_type, reward, discount, state = env.reset()
            frame_compressed = get_next_frame(env)


        # write data files so they can be used for eval
        if t % args.save_freq == 0:
            step_file_name = "{}_{}_{}_{:05d}_{:010d}".format(args.exp_name, args.policy, args.domain, args.seed, t)
            st = time.time()
            info['save_start_times'].append(st)
            print("---------------------------------------")
            step_file_path = os.path.join(results_dir, step_file_name)
            print("writing data files", step_file_name)
            # getting stuck here
            pickle.dump(replay_buffer, open(step_file_path+'.pkl', 'wb'))
            policy.save(step_file_path+'.pt')
            et = time.time()
            info['save_end_times'].append(et)
            utils.save_info_dict(info, step_file_path)
            print("finished writing files in {} secs".format(et-st))
 

def load_policy(load_model_path):
    # Initialize policy
    start_step = 0
    if args.policy == "TD3":
        import TD3
        # TODO does td3 give pos/neg since we only give max_action
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
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
        start_step = policy.step

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



 
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3", help='Policy name (TD3, DDPG or OurDDPG)')
    parser.add_argument("--domain", default="reacher", help='DeepMind Control Suite domain name')
    parser.add_argument("--task", default="easy", help='Deepmind Control Suite task name')
    parser.add_argument("--seed", default=0, type=int, help='random seed')
    parser.add_argument("--start_timesteps", default=25e3, type=int, help='number of time steps initial random policy is used')
    parser.add_argument("--replay_size", default=int(1e6), type=int, help='number of steps to store in replay buffer')
    parser.add_argument("--save_freq", default=50000, type=int, help='how often to save model and replay buffer')
    parser.add_argument("--eval_replay_size", default=int(500000), type=int, help='number of steps to store in replay buffer')
    parser.add_argument("-ne", "--num_eval_episodes", default=10, type=int, help='')
    parser.add_argument("--max_timesteps", default=1e7, type=int, help='max time steps to run environment')
    parser.add_argument("--expl_noise", default=0.1, help='std of Gaussian exploration noise')
    parser.add_argument("--batch_size", default=256, type=int, help='batch size for training agent')
    parser.add_argument("--discount", default=0.99, help='discount factor')
    parser.add_argument("--state_pixels", default=False, action='store_true', help='return pixels from cameras')                 # Discount factor
    parser.add_argument('-g', "--convert_to_gray", default=False, action='store_true', help='grayscale images')                 # Discount factor
    parser.add_argument("--frame_height", default=120)
    parser.add_argument("--frame_width", default=120)
    #parser.add_argument("--time_limit", default=10)                 # Time in seconds allowed to complete mujoco task
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int, help='freq of delayed policy updates')
    parser.add_argument("--n_bins", default=40, type=int)   


    parser.add_argument('-e', "--eval", default=False, action='store_true', help='evaluate')                 # Discount factor
    parser.add_argument('-ee', "--eval_all", default=False, action='store_true', help='evaluate all models in specified directory')                 # Discount factor

    parser.add_argument('-cv', '--camera_view', default='', help='camera view to use') 
    parser.add_argument('-d', '--device', default='cpu')   # use gpu rather than cpu for computation
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--load_replay", default="", help='Indicate replay buffer to load past experience from. Options are ["", "empty", file path of replay buffer.pkl]. If empty string, a new replay buffer will be created unless --load_model was invoked, in that case, the respective replay buffer will be loaded. If set to "empty", an new buffer will be created regardless of if --load_model was invoked.')
    parser.add_argument("--exp_name", default="test")               
    parser.add_argument("-ctn", "--continue_in_new_dir", default=False, action="store_true", help='If true, store results from loaded model in newly generated directory instead of resuming from loaded dir (possibly overwriting existing files).')               
    
    parser.add_argument("--savedir", default="results", help='overall dir to store checkpoints')               

    args = parser.parse_args()

    if args.eval or args.eval_all:
        args.state_pixels = True
        
    environment_kwargs = {'flat_observation':True}

    num_steps = 0
    file_name = "{}_{}_{}_{:05d}".format(args.exp_name, args.policy, args.domain, args.seed)
    print("---------------------------------------")
    print("Policy: {} Domain: {}, Task: {}, Seed: {}".format(args.policy, args.domain, args.task, args.seed))
    print("---------------------------------------")

    env = suite.load(domain_name=args.domain, task_name=args.task, task_kwargs={'random':args.seed},  environment_kwargs=environment_kwargs)

    # Set seeds
    torch.manual_seed(args.seed)
    if not args.state_pixels:
        cam_dim = [0,0,0]
    else:
        if args.convert_to_gray:
            cam_dim = [args.frame_height, args.frame_width, 1]
        else:
            cam_dim = [args.frame_height, args.frame_width, 3]

    state_dim = env.observation_spec()['observations'].shape[0]
    action_dim = env.action_spec().shape[0]


    min_action = float(env.action_spec().minimum[0])
    max_action = float(env.action_spec().maximum[0])
    action_shape = env.action_spec().shape
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "device":args.device,
    }


    if args.eval_all:
        assert os.path.isdir(args.load_model); print('--load_model must be an experiment directory if --eval_all')
        model_files = sorted(glob(os.path.join(args.load_model, '*.pt')))
        for xx, mf in enumerate(model_files):
            eval_buffer = mf.replace('.pt', '.epkl')
            if not os.path.exists(eval_buffer):
                policy, train_step, load_model_path, results_dir = load_policy(mf)
                evaluate()
 
    elif args.eval:
        policy, train_step, load_model_path, results_dir = load_policy(args.load_model)
        evaluate()
    else:
        policy, start_step, load_model_path, results_dir = load_policy(args.load_model)
        train(start_step)





