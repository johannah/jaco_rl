import numpy as np
import argparse
import os
import sys
import pickle
import utils

from dm_control import suite
from skimage.color import rgb2gray
from IPython import embed

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="jaco")       			# DeepMind Control Suite domain name
    parser.add_argument("--task", default="easy")  					# DeepMind Control Suite task name
    parser.add_argument("--seed", default=0, type=int)              # Sets PyTorch and Numpy seeds
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--max_timesteps", default=1e7, type=int)   # Max time steps to run environment
    parser.add_argument("--write_freq", default=1000, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--joint_states_file", default="../test_dataset/2020-03-10-15-24-35_joint_state.npz")   # actions load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()
    #environment_kwargs = {'flat_observation': False}
    environment_kwargs = {'flat_observation': True}
    task_kwargs = {'time_limit':25}
    args.policy = 'imitate_robot'

    robot_joint_states = np.load(args.joint_states_file)
    total_robot_steps = len(robot_joint_states['joint_secs'])
    num_steps = 0
    file_name = "{}_{}_{:05d}".format(args.policy, args.domain, args.seed)
    print("---------------------------------------")
    print("Policy: {} Domain: {}, Task: {}, Seed: {}".format(args.policy, args.domain, args.task, args.seed))
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    env = suite.load(domain_name=args.domain, task_name=args.task, task_kwargs=task_kwargs, environment_kwargs=environment_kwargs)

    # Set seeds
    np.random.seed(args.seed)

    state_dim = env.observation_spec()['observations'].shape[0]
    action_dim = env.action_spec().shape[0]
    # we will convert to gray
    cam_dim = list(env.physics.render().shape)[:2]
    min_action = float(env.action_spec().minimum[0])
    max_action = float(env.action_spec().maximum[0])
    action_shape = env.action_spec().shape
    print('total', total_robot_steps)

    base_action = np.random.uniform(low=min_action, high=max_action, size=action_shape)

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=int(100000), cam_dim=cam_dim)

    state_type, reward, discount, state = env.reset()


    for t in range(int(total_robot_steps)):
        action = robot_joint_states['joint_velocities'][t,:action_dim]
        # Perform action
        step_type, reward, discount, next_state = env.step(action)
        done = step_type.last()
        # Store data in replay buffer
        frame = (rgb2gray(env.physics.render())*255).astype(np.uint8)
        replay_buffer.add(state['observations'], action, next_state['observations'], reward, done, frame=frame)
        state = next_state

    step_file_name = "{}_{}_{:05d}_{:010d}".format(args.policy, args.domain, args.seed, t)
    print("writing data files", step_file_name)
    pickle.dump(replay_buffer, open("./results/{}.pkl".format(step_file_name), 'wb'))
