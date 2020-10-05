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
from replay_buffer import ReplayBuffer, compress_frame
import plotting 
import time
from glob import glob
from robot_physics import RobotPhysics
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from dm_control import suite
from dm_control import viewer
from IPython import embed

robot_frame_height = 640 
robot_frame_width = 480 

def get_next_frame(frame_env, height, width):
    if args.camera_view == 'double':
        next_frame_def = frame_env.physics.render(height=height,width=width,camera_id=-1)
        next_frame_top = frame_env.physics.render(height=height,width=width,camera_id='topview')
        next_frame = np.hstack((next_frame_def, next_frame_top))
        next_frame_compressed = compress_frame(next_frame)
    else:
        next_frame = frame_env.physics.render(height=height,width=width,camera_id=args.camera_view)
        print('framesize',  next_frame.shape, frame_env.physics.type)
        # 640x480 robot
        # 480x640 mujoco
        #plt.figure()
        #plt.imshow(next_frame)
        #plt.savefig('example.png')        
        #plt.close()
        if args.convert_to_gray:
            next_frame = img_as_ubyte(rgb2gray(next_frame)[:,:,None])
        next_frame_compressed = compress_frame(next_frame)
    return next_frame_compressed, next_frame

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
 
def mirror(load_model_filepath):
    use_robot = True
    print("starting evaluation for {} episodes".format(args.num_eval_episodes))
    policy, train_step, results_dir, loaded_modelpath = load_policy(load_model_filepath)
    eval_seed = args.seed+train_step
    task_kwargs['random'] = eval_seed
    load_model_base = loaded_modelpath.replace('.pt', '')
    state_names_dict = get_state_names_dict()
    eval_env = suite.load(domain_name=args.domain, task_name=args.task, task_kwargs=task_kwargs,  environment_kwargs=environment_kwargs)
    if use_robot: 
        robot_task_kwargs = task_kwargs
        robot_task_kwargs['physics_type'] = 'robot'
        robot_env = suite.load(domain_name=args.domain, task_name=args.task, task_kwargs=robot_task_kwargs,  environment_kwargs=environment_kwargs)

    # generate random seed
    random_state = np.random.RandomState(eval_seed)

    eval_dir = os.path.join(load_model_base + '_eval_CAM%s_S%s_S%05d_E%04d'%(args.camera_view, args.eval_filename_modifier, eval_seed, args.num_eval_episodes))
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    print('saving results to dir: {}'.format(eval_dir))
    eval_base = os.path.join(eval_dir, get_step_filename(train_step)+'_mirror_CAM%s_S%s_S%05d_E%04d'%(args.camera_view, args.eval_filename_modifier, eval_seed, args.num_eval_episodes))

    eval_step_filepath = eval_base + '.epkl' 
    eval_replay_buffer = ReplayBuffer(kwargs['state_dim'], kwargs['action_dim'], 
                                 max_size=int(args.eval_replay_size), 
                                 cam_dim=cam_dim, seed=eval_seed)
    if use_robot:
        robot_step_filepath = eval_base + '.rpkl' 
        robot_replay_buffer = ReplayBuffer(kwargs['state_dim'], kwargs['action_dim'], 
                                     max_size=int(args.eval_replay_size), 
                                     cam_dim=robot_cam_dim, seed=eval_seed)

    for e in range(args.num_eval_episodes):
        done = False
        num_steps = 0
        robot_num_steps = 0
        print('reset from mirror')
        state_type, reward, discount, state = eval_env.reset()
        if use_robot:
            robot_state_type, robot_reward, robot_discount, robot_state = robot_env.reset()

        frame_compressed, frame = get_next_frame(eval_env, args.frame_height, args.frame_width)
        if use_robot:
            robot_frame_compressed, robot_frame = get_next_frame(robot_env, robot_frame_height, robot_frame_width)
        while done == False:
            try:
                action = (
                        policy.select_action(state['observations'])
                    ).clip(-kwargs['max_action'], kwargs['max_action'])
                step_type, reward, discount, next_state = eval_env.step(action)
                next_frame_compressed, next_frame = get_next_frame(eval_env, args.frame_height, args.frame_width)
                print(num_steps, "----obs---")
                eo = np.array(state['observations'])
                if not use_robot:
                    if reward > .9:
                        done = True
                else:
                    ro = np.array(robot_state['observations'])%(2*np.pi)
                    # handle robot
                    robot_action = (
                            policy.select_action(robot_state['observations'])
                        ).clip(-kwargs['max_action'], kwargs['max_action'])

                    proposed_action = robot_state['observations'][3:3+7] + action
                    if robot_env.task.count_safety_violations(proposed_action):
                        robot_num_steps+=1
                        robot_step_type, robot_reward, robot_discount, robot_next_state = robot_env.step(action)
                        robot_next_frame_compressed, robot_next_frame = get_next_frame(robot_env, robot_frame_height, robot_frame_width)
                        if robot_reward > .9:
                            done = True
 
                        robot_replay_buffer.add(robot_state['observations'], robot_action, robot_reward, 
                                      robot_next_state['observations'], done, 
                                      frame_compressed=robot_frame_compressed, 
                                      next_frame_compressed=robot_next_frame_compressed)

                    else:
                        print('!!!! blocking unsafe robot step')

                eval_replay_buffer.add(state['observations'], action, reward, 
                                  next_state['observations'], done, 
                                  frame_compressed=frame_compressed, 
                                  next_frame_compressed=next_frame_compressed)

                frame_compressed = next_frame_compressed
                state = next_state
                frame = next_frame

                if use_robot:
                    robot_frame_compressed = robot_next_frame_compressed
                    robot_state = robot_next_state
                    robot_frame = robot_next_frame
                num_steps+=1
                time.sleep(1)
            except KeyboardInterrupt:
                embed()
 
        ## plot episode
        epath = eval_base+ '_E{}'.format(e)
        er = np.int(eval_replay_buffer.episode_rewards[-1])
        plotting.plot_states(eval_replay_buffer.get_last_steps(num_steps), epath, detail_dict=state_names_dict)
        plotting.plot_position_actions(eval_replay_buffer.get_last_steps(num_steps), epath, relative=True)
        pickle.dump(eval_replay_buffer, open(epath+'.epkl', 'wb'))

        if use_robot:
            rer = np.int(robot_replay_buffer.episode_rewards[-1])
            repath = eval_base+ '_E{}_robot'.format(e)
            plotting.plot_states(robot_replay_buffer.get_last_steps(num_steps), repath, detail_dict=state_names_dict)
            plotting.plot_position_actions(robot_replay_buffer.get_last_steps(robot_num_steps), repath, relative=True)
            pickle.dump(robot_replay_buffer, open(repath+'.rpkl', 'wb'))
        if np.max([args.plot_movie, args.plot_action_movie, args.plot_frames]):
            emovie_path = epath+'CAM{}.mp4'.format(e, er, args.camera_view)
            print('plotting episode: {}'.format(emovie_path))
            plotting.plot_frames(emovie_path, 
                                 eval_replay_buffer.get_last_steps(num_steps),
                                 plot_action_frames=args.plot_action_movie,
                                 min_action=-kwargs['max_action'], max_action=kwargs['max_action'],
                                 plot_frames=args.plot_frames)
            if use_robot:
                removie_path = epath+'ROBOT{}.mp4'.format(e, rer, args.camera_view)
                plotting.plot_frames(removie_path, 
                                     robot_replay_buffer.get_last_steps(robot_num_steps),
                                     plot_action_frames=args.plot_action_movie,
                                     min_action=-kwargs['max_action'], max_action=kwargs['max_action'],
                                 plot_frames=args.plot_frames)
  
    eval_replay_buffer.shrink_to_last_step()
    pickle.dump(eval_replay_buffer, open(eval_step_filepath, 'wb'))
    # plot evaluation
    plotting.plot_replay_reward(eval_replay_buffer, eval_base, start_step=train_step, name_modifier='_eval')
    plotting.plot_states(eval_replay_buffer.get_last_steps(eval_replay_buffer.size), eval_base+'_eval', detail_dict=state_names_dict)

    if use_robot:
        robot_replay_buffer.shrink_to_last_step()
        pickle.dump(robot_replay_buffer, open(robot_step_filepath, 'wb'))
        plotting.plot_replay_reward(robot_replay_buffer, eval_base, start_step=train_step, name_modifier='_robot')
        plotting.plot_states(robot_replay_buffer.get_last_steps(robot_replay_buffer.size), eval_base+'_robot', detail_dict=state_names_dict)

    if np.max([args.plot_movie, args.plot_action_movie, args.plot_frames]):
        movie_path = eval_base+'_CAM{}.mp4'.format(args.camera_view)
        plotting.plot_frames(movie_path, eval_replay_buffer.get_last_steps(eval_replay_buffer.size), plot_action_frames=args.plot_action_movie, min_action=-kwargs['max_action'], max_action=kwargs['max_action'], plot_frames=args.plot_frames)

        if use_robot:
            rmovie_path = eval_base+'_ROBOT{}.mp4'.format(args.camera_view)
            plotting.plot_frames(rmovie_path, robot_replay_buffer.get_last_steps(robot_replay_buffer.size), plot_action_frames=args.plot_action_movie, min_action=-kwargs['max_action'], max_action=kwargs['max_action'], plot_frames=args.plot_frames)

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

def get_step_filename(t):
    return "{}_{}_{}_{:05d}_{:010d}".format(args.exp_name, args.policy, args.domain, args.seed, t)

def get_step_filepath(results_dir, t):
    return os.path.join(results_dir, get_step_filename(t))

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
    parser.add_argument('-d', '--device', default='cpu', help="device to use for pytorch computation (cuda:0 or cpu)")
    parser.add_argument("--savedir", default="results", help='overall dir to store checkpoints')               
    parser.add_argument("--exp_name", default="test", help="name of experiment directory")               
    parser.add_argument("--policy", default="TD3", help='Policy name (TD3, DDPG or OurDDPG)')
    parser.add_argument("--domain", default="jaco", help='DeepMind Control Suite domain name')
    parser.add_argument("--task", default="fixed_relative_position_reacher_7DOF", help='Deepmind Control Suite task name')
    #parser.add_argument("--task", default="configurable_reacher", help='Deepmind Control Suite task name')
    parser.add_argument("--expl_noise", default=0.1, help='std of Gaussian exploration noise')
    parser.add_argument("--batch_size", default=256, type=int, help='batch size for training agent')
    parser.add_argument("--discount", default=0.99, help='discount factor')

    # training details
    parser.add_argument("--seed", default=0, type=int, help='random seed')
    parser.add_argument("--start_timesteps", default=25e3, type=int, help='number of time steps initial random policy is used')
    parser.add_argument("--replay_size", default=int(2e6), type=int, help='number of steps to store in replay buffer')
    parser.add_argument("--save_freq", default=50000, type=int, help='how often to save model and replay buffer')
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
    parser.add_argument("--load_replay", default="", help='Indicate replay buffer to load past experience from. Options are ["", "empty", file path of replay buffer.pkl]. If empty string, a new replay buffer will be created unless --load_model was invoked, in that case, the respective replay buffer will be loaded. If set to "empty", an new buffer will be created regardless of if --load_model was invoked.')
    parser.add_argument('-efm', '--eval_filename_modifier', default='', help='helper to modify the filename for a unique evaluation')
    parser.add_argument('-owr', "--overwrite_replay", default=False, action='store_true', help='gather new eval replay experience even if there is already a .epkl file')
    parser.add_argument("--eval_replay_size", default=int(500000), type=int, help='number of steps to store in replay buffer')
    parser.add_argument("-ne", "--num_eval_episodes", default=1, type=int, help='')
    parser.add_argument("--state_pixels", default=False, action='store_true', help='return pixels from cameras for plotting')              
    parser.add_argument('-g', "--convert_to_gray", default=False, action='store_true', help='grayscale images')               
    parser.add_argument("--frame_height", default=320)
    parser.add_argument("--frame_width", default=240)
    parser.add_argument("-ftp", "--fixed_target_position", default="[-.2, -.5, .5]", type=str)
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

    if args.load_model == "" and args.load_replay == "":
        raise ValueError; print("--load_model or --load_replay required to evaluate")
    environment_kwargs = {'flat_observation':True}

    num_steps = 0
    print("---------------------------------------")
    print("Policy: {} Domain: {}, Task: {}, Seed: {}".format(args.policy, args.domain, args.task, args.seed))
    print("---------------------------------------")
    # info for particular task
    fixed_target_position = eval(args.fixed_target_position)
    assert len(fixed_target_position) == 3
    task_kwargs = {'fixed_target_position':fixed_target_position}
    #task_kwargs = {'target_type':'fixed', 'fixed_target_position':[-.3, -.3, .5], 'relative_step':True}
    if args.domain == 'jaco':
        if args.fence_name == 'jodesk':
            # .1f is too low - joint 4 hit sometimes!!!!
            task_kwargs['fence'] = {'x':(-.45,.45), 'y':(-1.0, .4), 'z':(.15, 1.2)}
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
    cam_dim = [0,0,0]
    robot_cam_dim = [robot_frame_height, robot_frame_width, 3]
    if args.convert_to_gray:
        cam_dim = [args.frame_height, args.frame_width, 1]
    # get topview and front view
    elif args.camera_view == 'double':
        cam_dim = [args.frame_height, args.frame_width*2, 3]
    else:
        cam_dim = [args.frame_height, args.frame_width, 3]
    # Set seeds
    torch.manual_seed(args.seed)
    mirror(args.load_model)





