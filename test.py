import numpy as np
import torch
import argparse
import os
import sys
import pickle
import utils
from replay_buffer import ReplayBuffer

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


#def equal_quantization_undo(n_bins, d_actions):
#    return ((d_actions.double()/n_bins)*2)-1

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="random")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--domain", default="jaco")       			# DeepMind Control Suite domain name
    parser.add_argument("--task", default="easy")  					# DeepMind Control Suite task name
    parser.add_argument("--seed", default=0, type=int)              # Sets PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4, type=int) # Time steps initial random policy is used
    parser.add_argument("--replay_size", default=500000, type=int) # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5000, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e7, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--state_pixels", default=True, action='store_true', help='return pixels from cameras')                 # Discount factor
    parser.add_argument("--convert_to_gray", default=True, action='store_true', help='grayscale images')                 # Discount factor
    parser.add_argument("--frame_height", default=80)
    parser.add_argument("--frame_width", default=80)
    #parser.add_argument("--time_limit", default=10)                 # Time in seconds allowed to complete mujoco task
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--n_bins", default=40, type=int)       # Frequency of delayed policy updates
    parser.add_argument('-c', "--cuda", default=False, type=bool)   # use gpu rather than cpu for computation
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--exp_name", default="test")                 # Model load file name, "" doesn't load, "default" uses file_name

    args = parser.parse_args()
    #environment_kwargs = {'flat_observation': False}
    #environment_kwargs = {'flat_observation': True, "time_limit":args.time_limit}
    environment_kwargs = {'flat_observation': True}
    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    num_steps = 0
    file_name = "{}_{}_{:05d}".format(args.policy, args.domain, args.seed)
    print("---------------------------------------")
    print("Policy: {} Domain: {}, Task: {}, Seed: {}".format(args.policy, args.domain, args.task, args.seed))
    print("---------------------------------------")

    results_dir = os.path.join('results', args.exp_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    env = suite.load(domain_name=args.domain, task_name=args.task, environment_kwargs=environment_kwargs)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
        "min_action": min_action,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "device":device,
    }
    # Initialize policy
    if args.policy == "TD3":
        import TD3
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

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(policy_file)

    if args.convert_to_gray:
        cam_dim = [args.frame_height, args.frame_width, 1]
    else:
        cam_dim = [args.frame_height, args.frame_width, 3]

    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=int(args.replay_size), cam_dim=cam_dim)

    ## Evaluate untrained policy
    #evaluations = [eval_policy(policy, args.domain, args.task, args.seed)]
    # position is relative position from starting point

    state_type, reward, discount, state = env.reset()

    done = False
    frame = None
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    best_reward = 0
    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = random_state.uniform(low=min_action, high=max_action, size=action_shape)
        else:
            action = policy.select_action(state['observations'])
        #        #+ random_state.normal(0, max_action * args.expl_noise, size=action_dim)
        #        #).clip(-max_action, max_action)

        action = equal_quantization(args.n_bins, np.abs(action))
        if random_state.rand() > .5:
            action = action*-1.0
        # Perform action
        step_type, reward, discount, next_state = env.step(action)
        frame = env.physics.render(height=args.frame_height,width=args.frame_width,camera_id='view1')
        if args.convert_to_gray:
            frame = img_as_ubyte(rgb2gray(frame)[:,:,None])
        done = step_type.last()
        # Store data in replay buffer
        replay_buffer.add(state['observations'], action, next_state['observations'], reward, done, frame=frame)

        state = next_state
        episode_reward += reward
        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)
        if reward > best_reward:
            best_reward = reward
        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            #print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(t+1, episode_num+1, episode_timesteps, episode_reward))
            print('finished train episode with last reward of {} best reward {}'.format(reward, best_reward))
            print("---------------------------------------")
            # Reset environment
            state_type, reward, discount, state = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            best_reward = 0

        if done:
            print("-------------DONE", t)
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            #print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(t+1, episode_num+1, episode_timesteps, episode_reward))
            print('finished train episode with last reward of {} best reward {}'.format(reward, best_reward))
            print('end', t)
            print('target', state['observations'][-7:-4])
            print('finger', state['observations'][-4:-1])
            print('dist', state['observations'][-1:])
            print("---------------------------------------")

            # Reset environment
            state_type, reward, discount, state = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            best_reward = 0

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            step_file_name = "{}_{}_{:05d}_{:010d}".format(args.policy, args.domain, args.seed, t)
            st = time.time()
            print("---------------------------------------")
            print("---------------------------------------")
            print("---------------------------------------")
            print("writing data files", step_file_name)
            evaluations.append(eval_policy(policy, args.domain, args.task, args.seed))
            pickle.dump(replay_buffer, open("./results/{}.pkl".format(step_file_name), 'wb'))
            np.save("./results/{}".format(step_file_name), evaluations)
            policy.save("./models/{}".format(step_file_name))
            et = time.time()
            print("finished writing files in {} secs".format(et-st))
 

