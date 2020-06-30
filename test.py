import numpy as np
import torch
import argparse
import os
import sys
import pickle
import utils
from replay_buffer import ReplayBuffer
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3", help='Policy name (TD3, DDPG or OurDDPG)')
    parser.add_argument("--domain", default="jaco", help='DeepMind Control Suite domain name')
    parser.add_argument("--task", default="relative_reacher_easy", help='Deepmind Control Suite task name')
    parser.add_argument("--seed", default=0, type=int, help='random seed')
    parser.add_argument("--start_timesteps", default=25e3, type=int, help='number of time steps initial random policy is used')
    parser.add_argument("--replay_size", default=1000000, type=int, help='number of steps to store in replay buffer')
    parser.add_argument("--save_freq", default=10000, type=int, help='how often to save model and replay buffer')
    parser.add_argument("--max_timesteps", default=1e7, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--state_pixels", default=False, action='store_true', help='return pixels from cameras')                 # Discount factor
    parser.add_argument("--convert_to_gray", default=True, action='store_true', help='grayscale images')                 # Discount factor
    parser.add_argument("--frame_height", default=80)
    parser.add_argument("--frame_width", default=80)
    #parser.add_argument("--time_limit", default=10)                 # Time in seconds allowed to complete mujoco task
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int, help='freq of delayed policy updates')
    parser.add_argument("--n_bins", default=40, type=int)   



    parser.add_argument('-d', '--device', default='cpu')   # use gpu rather than cpu for computation
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

    env = suite.load(domain_name=args.domain, task_name=args.task, task_kwargs={'random_seed':args.seed},  environment_kwargs=environment_kwargs)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
    random_state = np.random.RandomState(args.seed)
    action = random_state.uniform(low=min_action, high=max_action, size=action_shape)
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "device":args.device,
    }
    # Initialize policy
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
    load_model_path = args.load_model
    if load_model_path != "":
        if os.path.isdir(args.load_model):
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
        policy.load(load_model_path)
        # store in old dir
        if not args.continue_in_new_dir:
            results_dir = os.path.split(load_model_path)[0]
            print("continuing in loaded directory")
            print(results_dir)
        else:
            print("resuming in new directory")
            print(results_dir)
        try:
            info = load_info_dict(load_model_path.replace('.pt', '.info'))
        except:
            print('---not able to load info path')
            embed()

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print('storing results in: {}'.format(results_dir))



    # load replay buffer
    load_replay_path = args.load_replay
    if args.load_replay == 'empty' or (args.load_replay == '' and args.load_model == '' ):
        replay_buffer = ReplayBuffer(state_dim, action_dim, 
                                     max_size=int(args.replay_size), 
                                     cam_dim=cam_dim, seed=args.seed)
    else:
        if args.load_replay != '':
            if os.path.isdir(args.load_replay):
                print("searching for latest replay from dir: {}".format(args.load_replay))
                # find last file
                search_path = os.path.join(args.load_replay, '*.pkl')
                replay_files = glob(search_path)
                if not len(replay_files):
                    raise ValueError('could not find replay files at {}'.format(search_path))
                else:
                    load_replay_path = sorted(replay_files)[-1]
                    print('loading most recent replay from directory: {}'.format(load_replay_path))
            else:
                load_replay_path = args.load_replay
        else:
            load_replay_path = load_model_path.replace('.pt', '.pkl')
        print("loading replay from: {}".format(load_replay_path))
        replay_buffer = pickle.load(open(load_replay_path, 'rb'))

    info = utils.create_new_info_dict(args, load_model_path, load_replay_path)
    start_t = replay_buffer.size
    done = False
    frame = None
    state_type, reward, discount, state = env.reset()
    for t in range(start_t, int(args.max_timesteps)):
        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = random_state.uniform(low=min_action, high=max_action, size=action_shape)
        else:
            # Select action randomly or according to policy
            if t < args.start_timesteps:
                action = np.random.uniform(low=min_action, high=max_action, size=action_dim)
            else:
                action = (
                    policy.select_action(state['observations'])
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                ).clip(-max_action, max_action)
        # Perform action
        step_type, reward, discount, next_state = env.step(action)
        if args.state_pixels:
            frame = env.physics.render(height=args.frame_height,width=args.frame_width,camera_id='topview')
            if args.convert_to_gray:
                frame = img_as_ubyte(rgb2gray(frame)[:,:,None])
        done = step_type.last()
        # Store data in replay buffer
        replay_buffer.add(state['observations'], action, next_state['observations'], reward, done, frame=frame)
        state = next_state
        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(t, replay_buffer, args.batch_size)
        if done:
            print("Total T: {} Episode Num: {} Reward: {}".format(t, replay_buffer.episode_count-1, replay_buffer.episode_rewards[-1]))
            print("---------------------------------------")
            # Reset environment
            state_type, reward, discount, state = env.reset()
        # write data files so they can be used for eval
        if (t + 1) % args.save_freq == 0:
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
 

