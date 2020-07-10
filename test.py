import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from copy import deepcopy
import numpy as np
import torch
import argparse
import os
import sys
import pickle
import utils
from replay_buffer import ReplayBuffer, compress_frame, plot_frames, plot_replay_reward, plot_states, plot_position_actions
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
        next_frame = frame_env.physics.render(height=args.frame_height,width=args.frame_width,camera_id=args.camera_view)
        if args.convert_to_gray:
            next_frame = img_as_ubyte(rgb2gray(next_frame)[:,:,None])
        next_frame = compress_frame(next_frame)
    return next_frame

def plot_loss_dict(policy, load_model_base):
    loss_plot_path = load_model_base + '_loss.png'
    loss_dict = policy.get_loss_plot_data()
    plt.figure()
    for key, val in loss_dict.items():
        plt.plot(val[0], val[1], label=key)
    plt.title('Training Loss')
    plt.legend(loc=2)
    plt.savefig(loss_plot_path)
    plt.close()
    
def get_state_names_dict():
    plot_environment_kwargs = deepcopy(environment_kwargs)
    plot_environment_kwargs['flat_observation'] = False
    _plot_env = suite.load(domain_name=args.domain, task_name=args.task, task_kwargs=task_kwargs,  environment_kwargs=plot_environment_kwargs)
    plot_spec = _plot_env.observation_spec()
    del _plot_env; del plot_environment_kwargs
    state_names_dict = {}
    st = 0
    for key in plot_spec.keys():
        for ind in range(plot_spec[key].shape[0]):
            state_names_dict[key+'_%02d'%ind] = np.arange(st+ind, st+ind+1)
        st = st+ind+1
    return state_names_dict
 
def evaluate(load_model_filepath):
    print("starting evaluation for {} episodes".format(args.num_eval_episodes))
    policy, train_step, results_dir, loaded_modelpath = load_policy(load_model_filepath)
    eval_seed = args.seed+train_step
    task_kwargs['random'] = eval_seed
    load_model_base = loaded_modelpath.replace('.pt', '')
    plot_loss_dict(policy, load_model_base)
    # TODO this isn't right - need to track steps in replay really

    state_names_dict = get_state_names_dict()
    train_replay_buffer = load_replay_buffer(load_model_base + '.pkl')

    eval_env = suite.load(domain_name=args.domain, task_name=args.task, task_kwargs=task_kwargs,  environment_kwargs=environment_kwargs)

    # generate random seed
    random_state = np.random.RandomState(eval_seed)
    train_dir = os.path.join(load_model_base + '_train%s'%args.eval_filename_modifier)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    train_base = os.path.join(train_dir, get_step_filename(train_step)+'_train')
    plot_replay_reward(train_replay_buffer, train_base, start_step=train_step, name_modifier='train')
    plot_states(train_replay_buffer.get_last_steps(train_replay_buffer.size), 
                train_base, detail_dict=state_names_dict)
 

    eval_dir = os.path.join(load_model_base + '_eval%s'%args.eval_filename_modifier)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    print('saving results to dir: {}'.format(eval_dir))
    eval_base = os.path.join(eval_dir, get_step_filename(train_step)+'_eval{:05d}'.format(eval_seed))


    eval_step_filepath = load_model_base + '.epkl'
    if os.path.exists(eval_step_filepath) and not args.overwrite_replay:
        print('loading existing replay buffer:{}'.format(eval_step_filepath))
        eval_replay_buffer = load_replay_buffer(eval_step_filepath)
    else:

        eval_replay_buffer = ReplayBuffer(kwargs['state_dim'], kwargs['action_dim'], 
                                     max_size=int(args.eval_replay_size), 
                                     cam_dim=cam_dim, seed=eval_seed)
 

        for e in range(args.num_eval_episodes):
            done = False
            num_steps = 0
            state_type, reward, discount, state = eval_env.reset()
            frame_compressed = get_next_frame(eval_env)
            # TODO off by one error in step count!? of replay_buffer
            while done == False:
                action = (
                        policy.select_action(state['observations'])
                    ).clip(-kwargs['max_action'], kwargs['max_action'])
                #print('E{}N{}'.format(e,num_steps))

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
 

            # plot episode
            er = np.int(eval_replay_buffer.episode_rewards[-1])
            epath = eval_base+ '_E{}_R{}'.format(e, er)
            exp = eval_replay_buffer.get_last_steps(num_steps)
            plot_states(exp, epath, detail_dict=state_names_dict)
            if args.domain == 'jaco':
                plot_position_actions(exp, epath, relative=True)
            emovie_path = epath+'CAM{}.mp4'.format(e, er, args.camera_view)
            plot_frames(emovie_path, eval_replay_buffer.get_last_steps(num_steps), plot_action_frames=args.plot_action_movie, min_action=-kwargs['max_action'], max_action=kwargs['max_action'], plot_frames=args.plot_frames)
        # write data files
        print("---------------------------------------")
        eval_replay_buffer.shrink_to_last_step()
        pickle.dump(eval_replay_buffer, open(eval_step_filepath, 'wb'))

    plot_replay_reward(eval_replay_buffer, eval_base, start_step=train_step, name_modifier='eval')
    plot_states(eval_replay_buffer.get_last_steps(eval_replay_buffer.size), 
                eval_base, detail_dict=state_names_dict)

    if args.plot_movie:
        movie_path = eval_base+'_CAM{}.mp4'.format( args.camera_view)
        plot_frames(movie_path, eval_replay_buffer.get_last_steps(eval_replay_buffer.size), plot_action_frames=args.plot_action_movie, min_action=-kwargs['max_action'], max_action=kwargs['max_action'], plot_frames=args.plot_frames)
    return eval_replay_buffer, eval_step_filepath

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

def get_step_filename(t):
    return "{}_{}_{}_{:05d}_{:010d}".format(args.exp_name, args.policy, args.domain, args.seed, t)

def get_step_filepath(results_dir, t):
    return os.path.join(results_dir, get_step_filename(t))

def train():
    task_kwargs['random'] = args.seed
    random_state = np.random.RandomState(task_kwargs['random'])
    policy, start_t, results_dir, loaded_modelpath = load_policy(args.load_model)
    env = suite.load(domain_name=args.domain, task_name=args.task, task_kwargs=task_kwargs,  environment_kwargs=environment_kwargs)
    #TODO - use action dim instead? 
    action_shape = env.action_spec().shape
    replay_buffer = load_replay_buffer(load_replay_path=args.load_replay, load_model_path=args.load_model, kwargs=kwargs, seed=args.seed)
    info = utils.create_new_info_dict(args, loaded_modelpath, args.load_replay)
    done = False
    state_type, reward, discount, state = env.reset()
    frame_compressed = get_next_frame(env)
    for t in range(start_t, int(args.max_timesteps)):
        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = random_state.uniform(low=-kwargs['max_action'], high=kwargs['max_action'], size=action_shape)
        else:
            # Select action randomly or according to policy
            action = (
                    policy.select_action(state['observations'])
                    + random_state.normal(0, kwargs['max_action'] * args.expl_noise, size=kwargs['action_dim'])
                ).clip(-kwargs['max_action'], kwargs['max_action'])
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
            st = time.time()
            info['save_start_times'].append(st)
            print("---------------------------------------")
            step_filepath = get_step_filepath(results_dir, t)
            print("writing data files", step_filepath)
            # getting stuck here
            pickle.dump(replay_buffer, open(step_filepath+'.pkl', 'wb'))
            policy.save(step_filepath+'.pt')
            et = time.time()
            info['save_end_times'].append(et)
            utils.save_info_dict(info, step_filepath)
            print("finished writing files in {} secs".format(et-st))
 

def load_policy(load_from):
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
    load_model_path = ''
    results_dir = os.path.join(args.savedir, args.exp_name+'%02d'%exp_cnt)
    while os.path.exists(results_dir):
        exp_cnt+=1
        results_dir = os.path.join(args.savedir, args.exp_name+'%02d'%exp_cnt)

    # load model if necessary
    if load_from != "":
        if os.path.isdir(load_from):
            print("loading latest model from dir: {}".format(load_from))
            # find last file
            search_path = os.path.join(load_from, '*.pt')
            model_files = glob(search_path)
            if not len(model_files):
                print('could not find model exp files at {}'.format(search_path))
                raise
            else:
                load_model_path = sorted(model_files)[-1]
        else:
            load_model_path = load_from
            print("loading model from file: {}".format(load_model_path))
        policy.load(load_model_path)
        # TODO 
        # utils.load_info_dict(load_model_base)
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
    return policy, start_step, results_dir, load_model_path

def get_kwargs(env):
    state_dim = env.observation_spec()['observations'].shape[0]
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
    parser.add_argument("--domain", default="reacher", help='DeepMind Control Suite domain name')
    parser.add_argument("--task", default="easy", help='Deepmind Control Suite task name')
    parser.add_argument("--seed", default=0, type=int, help='random seed')
    parser.add_argument("--use_robot", default=False, action='store_true')
    parser.add_argument('-owr', "--overwrite_replay", default=False, action='store_true', help='gather new eval replay experience even if there is already a .pkl file')
    parser.add_argument("--start_timesteps", default=25e3, type=int, help='number of time steps initial random policy is used')
    parser.add_argument("--replay_size", default=int(2e6), type=int, help='number of steps to store in replay buffer')
    parser.add_argument("--save_freq", default=50000, type=int, help='how often to save model and replay buffer')
    parser.add_argument("--eval_replay_size", default=int(500000), type=int, help='number of steps to store in replay buffer')
    parser.add_argument("-ne", "--num_eval_episodes", default=10, type=int, help='')
    parser.add_argument("--max_timesteps", default=1e7, type=int, help='max time steps to run environment')
    parser.add_argument("--expl_noise", default=0.1, help='std of Gaussian exploration noise')
    parser.add_argument("--batch_size", default=256, type=int, help='batch size for training agent')
    parser.add_argument("--discount", default=0.99, help='discount factor')
    parser.add_argument("--state_pixels", default=False, action='store_true', help='return pixels from cameras')                 # Discount factor
    parser.add_argument('-g', "--convert_to_gray", default=False, action='store_true', help='grayscale images')                 # Discount factor
    parser.add_argument("--frame_height", default=200)
    parser.add_argument("--frame_width", default=240)
    #parser.add_argument("--time_limit", default=10)                 # Time in seconds allowed to complete mujoco task
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int, help='freq of delayed policy updates')
    parser.add_argument("--n_bins", default=40, type=int)   


    parser.add_argument('-e', "--eval", default=False, action='store_true', help='evaluate')                 
    parser.add_argument('-pm', "--plot_movie", default=False, action='store_true', help='write a movie of episodes')                 
    parser.add_argument('-pam', "--plot_action_movie", default=False, action='store_true', help='write a movie with state view, actions, rewards, and next state views')                 
    parser.add_argument('-pf', "--plot_frames", default=False, action='store_true', help='write a movie and individual frames')                 
    parser.add_argument('-ee', "--eval_all", default=False, action='store_true', help='evaluate all models in specified directory')                 # Discount factor

    parser.add_argument('-cv', '--camera_view', default=-1, help='camera view to use. -1 will use the default view.') 
    parser.add_argument('-efm', '--eval_filename_modifier', default='')
    parser.add_argument('-d', '--device', default='cpu')   # use gpu rather than cpu for computation
    parser.add_argument('-fn', '--fence_name', default='jodesk', help='virtual fence name that prevents jaco from leaving this region.')   # use gpu rather than cpu for computation
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses filename
    parser.add_argument("--load_replay", default="", help='Indicate replay buffer to load past experience from. Options are ["", "empty", file path of replay buffer.pkl]. If empty string, a new replay buffer will be created unless --load_model was invoked, in that case, the respective replay buffer will be loaded. If set to "empty", an new buffer will be created regardless of if --load_model was invoked.')
    parser.add_argument("--exp_name", default="test")               
    parser.add_argument("-ctn", "--continue_in_new_dir", default=False, action="store_true", help='If true, store results from loaded model in newly generated directory instead of resuming from loaded dir (possibly overwriting existing files).')               
    
    parser.add_argument("--savedir", default="results", help='overall dir to store checkpoints')               

    args = parser.parse_args()

    #if args.eval or args.eval_all:
    #    args.state_pixels = True
        
    environment_kwargs = {'flat_observation':True}

    num_steps = 0
    print("---------------------------------------")
    print("Policy: {} Domain: {}, Task: {}, Seed: {}".format(args.policy, args.domain, args.task, args.seed))
    print("---------------------------------------")
    # info for particular task
    task_kwargs = {}
    if args.domain == 'jaco':
        task_kwargs = {'xml_name':"jaco_j2s7s300_position.xml", 
                       'start_position':'home', 
                       'relative_step':True, 
                       'target_type':'random',
                       'physics_type':'mujoco'}


        if args.fence_name == 'jodesk':
            # .1f is too low - joint 4 hit sometimes!!!!
            task_kwargs['fence'] = {'x':(-.5,.5), 'y':(-1.0, .4), 'z':(.15, 1.2)}
        else:
            task_kwargs['fence'] = {'x':(-5,5), 'y':(-5, 5), 'z':(.15, 1.2)}
        if args.use_robot:
            task_kwargs['physics_type'] = 'robot'
            args.eval_filename_modifier += 'robot'
        else:
            task_kwargs['physics_type'] = 'mujoco'

    _env = suite.load(domain_name=args.domain, task_name=args.task, task_kwargs=task_kwargs,  environment_kwargs=environment_kwargs)
    kwargs = get_kwargs(_env)
    del _env

    if args.eval:
        args.state_pixels = True
    if not args.state_pixels:
        cam_dim = [0,0,0]
    else:
        ## -1 is default camera view, but we also allow named cameras
        #try:
        #    args.camera_view = int(args.camera_view)
        #except:
        #    pass
            
        if args.convert_to_gray:
            cam_dim = [args.frame_height, args.frame_width, 1]
        else:
            cam_dim = [args.frame_height, args.frame_width, 3]
    # Set seeds
    torch.manual_seed(args.seed)
    if args.eval_all:
        assert os.path.isdir(args.load_model); print('--load_model must be an experiment directory if --eval_all')
        model_files = sorted(glob(os.path.join(args.load_model, '*.pt')))
        for xx, mf in enumerate(model_files):
            eval_replay_buffer, eval_step_filepath = evaluate(mf)
    elif args.eval:
        eval_replay_buffer, eval_step_filepath = evaluate(args.load_model)
    else:
        train()





