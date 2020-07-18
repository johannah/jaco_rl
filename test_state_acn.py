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
import time
from glob import glob

from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from dm_control import suite
from dm_control import viewer
from IPython import embed

import utils
import plotting 
from replay_buffer import ReplayBuffer, compress_frame
from train_state_acn import ACNresAngle
from functions import vq, vq_st
from acn_models import tPTPriorNetwork
from skvideo.io import vwrite
import zlib

def get_step_filename(t):
    return "{}_{}_{}_{:05d}_{:010d}".format(args.exp_name, args.policy, args.domain, args.seed, t)

def get_step_filepath(results_dir, t):
    return os.path.join(results_dir, get_step_filename(t))

def decompress_frame(z):
    return np.frombuffer(zlib.decompress(z), dtype=np.uint8).reshape(cam_dim)

def get_next_frame(frame_env):
    next_frame = None
    if args.state_pixels:
        next_frame = frame_env.physics.render(height=args.frame_height,width=args.frame_width,camera_id=args.camera_view)
        if args.convert_to_gray:
            next_frame = img_as_ubyte(rgb2gray(next_frame)[:,:,None])
        next_frame = compress_frame(next_frame)
    return next_frame

def get_latent(state_diff):
    rel_st = torch.FloatTensor(state_diff).to(args.device)
    state = torch.cat((rel_st,rel_st,rel_st,rel_st),3)
    u_q = model(state)
    u_q_flat = u_q.view(1, code_length).contiguous()
    u_p, s_p = prior_model(u_q_flat)
    rec_st, z_e_x, z_q_x, latents = model.decode(u_q)
    neighbor_distances, neighbor_indexes = prior_model.kneighbors(u_q_flat, n_neighbors=num_k)
    return rec_st, neighbor_indexes

def plot_latent_estimates(load_model_filepath):
    print("starting evaluation for {} episodes".format(args.num_eval_episodes))
    policy, train_step, results_dir, loaded_modelpath = load_policy(load_model_filepath)
    eval_seed = args.seed+train_step
    task_kwargs['random'] = eval_seed
    load_model_base = loaded_modelpath.replace('.pt', '')
    random_state = np.random.RandomState(eval_seed)

    action_env = suite.load(domain_name=args.domain, task_name=args.task, task_kwargs=task_kwargs,  environment_kwargs=environment_kwargs)
    diff_env = suite.load(domain_name=args.domain, task_name=args.task, task_kwargs=task_kwargs,  environment_kwargs=environment_kwargs)
    latent_env = suite.load(domain_name=args.domain, task_name=args.task, task_kwargs=task_kwargs,  environment_kwargs=environment_kwargs)
    # generate random seed

    latent_dir = os.path.join(load_model_base + '_swap_latent%s'%args.eval_filename_modifier)
    if not os.path.exists(latent_dir):
        os.makedirs(latent_dir)
    print('saving results to dir: {}'.format(latent_dir))
    latent_base = os.path.join(latent_dir, get_step_filename(train_step)+'_latent_S{:05d}'.format(eval_seed))

    action_step_filepath = latent_base + '_action_%s.epkl'%args.eval_filename_modifier
    diff_step_filepath = latent_base + '_diff_%s.epkl'%args.eval_filename_modifier
    latent_step_filepath = latent_base + '_action_%s.epkl'%args.eval_filename_modifier
    for e in range(args.num_eval_episodes):
        done = False
        num_steps = 0

        print('action')
        state_type, reward, discount, state = action_env.reset()
        frame_compressed = get_next_frame(action_env)
        print('diff')
        diff_env.reset()
        print('latent')
        latent_env.reset()
 
        frames = []
        while done == False:
            action = (
                    policy.select_action(state['observations'])
                ).clip(-kwargs['max_action'], kwargs['max_action'])
            # Perform action
            step_type, reward, discount, next_state = action_env.step(action)
            next_frame_compressed = get_next_frame(action_env)
            done = step_type.last()

            state_diff = (next_state['observations']-state['observations'])[None,None,None,3:10]
            rec_state_diff, neighbor_indexes = get_latent(state_diff)
            rec_state_diff = rec_state_diff.detach().cpu().numpy()

            _, _, _, diff_next_state = diff_env.step(state_diff[0,0,0])
            diff_next_frame_compressed = get_next_frame(diff_env)
            _, _, _, latent_next_state = latent_env.step(rec_state_diff[0,0,0])
            latent_next_frame_compressed = get_next_frame(latent_env)
            
            frame_compressed = next_frame_compressed
            diff_frame_compressed = diff_next_frame_compressed
            latent_frame_compressed = latent_next_frame_compressed
                          
            state = next_state
            num_steps+=1
            frames.append(np.hstack((decompress_frame(frame_compressed), 
                           decompress_frame(diff_frame_compressed), 
                           decompress_frame(latent_frame_compressed))))
        vwrite(latent_base+'reconstruct_%02d.mp4'%e, np.array(frames))

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
    parser.add_argument('-d', '--device', default='cpu', help="device to use for pytorch computation (cuda:0 or cpu)")
    parser.add_argument("--savedir", default="results", help='overall dir to store checkpoints')               
    parser.add_argument("--exp_name", default="test", help="name of experiment directory")               
    parser.add_argument("--policy", default="TD3", help='Policy name (TD3, DDPG or OurDDPG)')
    parser.add_argument("--expl_noise", default=0.1, help='std of Gaussian exploration noise')
    parser.add_argument("--discount", default=0.99, help='discount factor')

    # training details
    parser.add_argument("--seed", default=0, type=int, help='random seed')
    parser.add_argument("--start_timesteps", default=25e3, type=int, help='number of time steps initial random policy is used')
    parser.add_argument("--max_timesteps", default=1e7, type=int, help='max time steps to train agent')
    parser.add_argument("--tau", default=0.005, help="Target network update rate")
    parser.add_argument("--policy_noise", default=0.2, help="Noise added to target policy during critic update")
    parser.add_argument("--noise_clip", default=0.5, help="Range to clip target policy noise")
    parser.add_argument("--policy_freq", default=2, type=int, help='freq of delayed policy updates')

    # JACO Specific: 
    parser.add_argument('-fn', '--fence_name', default='jodesk', help='virtual fence name that prevents jaco from leaving this region. Hard code fences below.')  
    # real robot config
    parser.add_argument("--use_robot", default=False, action='store_true', help="start real robot experiment - requires ros_interface to be running")

    # eval / plotting helpers
    parser.add_argument("--load_model", default="", help="load .pt latest model from this directory or specify exact file") 
    parser.add_argument('-efm', '--eval_filename_modifier', default='', help='helper to modify the filename for a unique evaluation')
    parser.add_argument("--eval_replay_size", default=int(500000), type=int, help='number of steps to store in replay buffer')
    parser.add_argument("-ne", "--num_eval_episodes", default=10, type=int, help='')
    parser.add_argument('-g', "--convert_to_gray", default=False, action='store_true', help='grayscale images')               
    parser.add_argument("--frame_height", default=200)
    parser.add_argument("--frame_width", default=240)
    parser.add_argument('-pm', "--plot_movie", default=False, action='store_true', help='write a movie of episodes')                 
    parser.add_argument('-pam', "--plot_action_movie", default=False, action='store_true', help='write a movie with state view, actions, rewards, and next state views')                 
    parser.add_argument('-pf', "--plot_frames", default=False, action='store_true', help='write a movie and individual frames')                 
    parser.add_argument('-cv', '--camera_view', default=-1, help='camera view to use. -1 will use the default view.') 

    parser.add_argument("-ctn", "--continue_in_new_dir", default=False, action="store_true", help='If true, store results from loaded model in newly generated directory instead of resuming from loaded dir (possibly overwriting existing files).')               


    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    args.state_pixels = True
    args.domain = "jaco"
    args.task = "relative_position_reacher_7DOF"
    train_buffer_fname = 'results/relpos00/relpos_TD3_jaco_00000_0000060000.pkl'
    acn_model_fpath = 'models/model_000061161984.pt'
    acn_replay_buffer = pickle.load(open(train_buffer_fname, 'rb'))
    batch_size = 128
    code_length = 112
    vq_commitment_beta = 0.25
    num_k = 5
    model = ACNresAngle().to(args.device)
    train_size = acn_replay_buffer.size
    print("setting up prior with training size {}".format(train_size))
    prior_model = tPTPriorNetwork(size_training_set=train_size, 
                                  code_length=code_length,
                                  k=num_k).to(args.device)
    prior_model.eval()
    model.eval()
    model_dict = torch.load(acn_model_fpath)
    model.load_state_dict(model_dict['model'])
    prior_model.load_state_dict(model_dict['prior_model'])

    environment_kwargs = {'flat_observation':True}
    num_steps = 0
    print("---------------------------------------")
    print("Policy: {} Domain: {}, Task: {}, Seed: {}".format(args.policy, args.domain, args.task, args.seed))
    print("---------------------------------------")
    # info for particular task
    task_kwargs = {}
    args.domain == 'jaco'
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

    # if we need to make a movie, must have frames
    if np.max([args.plot_movie, args.plot_action_movie, args.plot_frames]):
        args.state_pixels = True
    if not args.state_pixels:
        cam_dim = [0,0,0]
    else:
        if args.convert_to_gray:
            cam_dim = [args.frame_height, args.frame_width, 1]
        else:
            cam_dim = [args.frame_height, args.frame_width, 3]
    # Set seeds
    torch.manual_seed(args.seed)
    plot_latent_estimates(args.load_model)





