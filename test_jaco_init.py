import pickle
from dm_control import viewer
from dm_control import suite
from replay_buffer import ReplayBuffer
import numpy as np
import os
from IPython import embed
import six
seed = 13
exp_name = 'test_start'
frame_height = 240
frame_width = 320
colors = 3

for exp_name ['relative_reacher_easy', 'reacher_easy', 'reacher_medium']:
savedir = 'results'
results_dir = os.path.join(savedir, exp_name)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)
 
random_state = np.random.RandomState(seed)
env = suite.load(domain_name='jaco', task_name='relative_reacher_easy', 
                 environment_kwargs={'flat_observation':False})

cam_dim = [frame_height, frame_width, colors]
action_dim = env.action_spec().shape[0]
state_dim = env.observation_spec()
print('starting jaco with {} actions'.format(action_dim))
# different joints have different mins/maxes
min_actions = env.action_spec().minimum
max_actions = env.action_spec().maximum
print('min action: {} max_action: {}'.format(min_actions, max_actions))
env.reset()

def random_policy():
    return [float(random_state.uniform(low=min_actions[i], high=max_actions[i], size=1)) for i in range(action_dim)]

max_timesteps = 512 
replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=max_timesteps, cam_dim=cam_dim)

state_type, reward, discount, state = env.reset()
t = 0
action = [.15,0,-.1,0,0,0,0]
while t < max_timesteps:
    #action = np.abs(random_policy())
    step_type, reward, discount, next_state = env.step(action)
    done = step_type.last()
    frame = env.physics.render(height=frame_height,width=frame_width,camera_id='topview')
    replay_buffer.add(state, action, next_state, reward, done, frame=frame)
    state = next_state
    t+=1
    if done:
       state_type, reward, discount, state = env.reset()
       break

step_file_name = "{}_{}_{}_{:05d}_{:010d}".format(exp_name, 'random', 'jaco', seed, t)
step_file_path = os.path.join(results_dir, step_file_name)
pickle.dump(replay_buffer, open(step_file_path+'.pkl', 'wb'))
