import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import shutil
from skvideo.io import vwrite
import shutil
from IPython import embed

import numpy as np
np.set_printoptions(precision=3, suppress=True)

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
        
def plot_frames(movie_fpath, last_steps, plot_frames=False, plot_action_frames=True, min_action=-.8, max_action=.8, min_reward=-1, max_reward=1):
    st, ac, re, nst, nd, fr, nfr = last_steps
    if fr.shape[1] == 0:
        raise ValueError; print("invalid frame shape, run with --state_pixels to ensure frames are rendered")
    n_steps = ac.shape[0]
    if plot_action_frames:
        n_actions = ac.shape[1]
        n_bars = n_actions + 1
        _, hsize, fr_wsize, nc = fr.shape
        wsize = int(fr_wsize*.40)
        n_bins = wsize
        cp = n_bins//2
        hw = hsize//n_bars
        action_bins = np.linspace(min_action, max_action, n_bins)
        reward_bins = np.linspace(min_reward, max_reward, n_bins)
        canvas = np.zeros((n_steps, hsize, wsize+1, nc), dtype=np.uint8)
        viridis = cm.get_cmap('viridis', n_actions)
        action_colors = np.array([np.array(viridis(x)[:nc]) for x in  np.linspace(0, 1, n_actions)])
        action_colors = (action_colors*255).astype(np.uint8)
        # red rewward
        reward_color = np.array([255,0,0])[:nc]
        try:
            for s in range(n_steps):
                for an,av in enumerate(ac[s]):
                    b = np.digitize(av, action_bins, right=True)
                    c = action_colors[an]
                    inds = np.linspace(cp-(cp-b), cp, np.abs(cp-b)+1, dtype=np.int)
                    canvas[s,hw*an:(hw*an)+hw,inds] = c
                rb = np.digitize(re[s][0], reward_bins, right=True)
                rinds = np.linspace(cp-(cp-rb), cp, np.abs(cp-rb)+1, dtype=np.int)
                canvas[s,hw*(n_bars-1):(hw*n_bars),rinds] = reward_color
            output = np.concatenate((fr, canvas, nfr), 2)
            vwrite(movie_fpath, output)
        except Exception as e:
            print(e)
            embed()
    else:
        vwrite(movie_fpath, fr)
    if plot_frames:
        dir_path = movie_fpath.replace('.mp4', '')
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
        init = 3 + 13+13+13
        TDIS = np.arange(0,3)
        DH4 = np.arange(init, init+3)
        DH6 = np.arange(init+3, init+3+3)
        DHTP = np.arange(init+3+3, init+3+3+3)
        TP = np.arange(init+3+3+3, init+3+3+3+3)
        minit = init+3+3+3+3
        J4 = np.arange(minit, minit+3)
        J6 = np.arange(minit+3, minit+3+3)
        fing = np.arange(minit+3+3, minit+3+3+3)
        target = np.arange(minit+3+3+3, minit+3+3+3+3)
        
        for n in range(fr.shape[0]):
            f,ax = plt.subplots(1,2)
            ax[0].imshow(fr[n])
            ax[1].imshow(nfr[n])

            target_title = 'TAR:x{:.2f}y{:.2f}z{:.2f}'.format(st[n,target][0], st[n,target][1], st[n,target][2]) + '\nTP:x{:.2f}y{:.2f}z{:.2f}'.format(st[n,TP][0], st[n,TP][1], st[n,TP][2]) + '\nS{}DISx{:.2f}y{:.2f}z{:.2f}'.format(n, st[n,TDIS][0], st[n,TDIS][0], st[n,TDIS][0])

            dh4 =  'Dj4:x{:.2f}y{:.2f}z{:.2f}'.format(st[n,DH4][0], st[n,DH4][1], st[n,DH4][2]) 
            dh6 =  'Dj6:x{:.2f}y{:.2f}z{:.2f}'.format(st[n,DH6][0], st[n,DH6][1], st[n,DH6][2])            
            dhTP = 'DT :x{:.2f}y{:.2f}z{:.2f}'.format(st[n,DHTP][0], st[n,DHTP][1], st[n,DHTP][2])
            m4 =   'mj4:x{:.2f}y{:.2f}z{:.2f}'.format(st[n,J4][0], st[n,J4][1], st[n,J4][2])
            m6 =   'mj6:x{:.2f}y{:.2f}z{:.2f}'.format(st[n,J6][0], st[n,J6][1], st[n,J6][2]) 
            mTP =  'mTP:x{:.2f}y{:.2f}z{:.2f}'.format(st[n,fing][0], st[n,fing][1], st[n,fing][2])

            ax[0].set_title(target_title)
            ax[1].set_title(dh4+'\n'+m4+'\n'+dh6+'\n'+m6+'\n'+dhTP+'\n'+mTP)
            #ax[1].set_title(nst_title+"S:%s R:%s"%(n,re[n]))
            img_path = os.path.join(dir_path, 'frame_%05d.png'%n)
            if not n %20:
                print('writing', img_path)
            plt.savefig(img_path)
            plt.close()
        cmd = "ffmpeg -pattern_type glob -i '%s' -c:v libx264 '%s' -y"%(os.path.join(dir_path, '*.png'), os.path.join(dir_path, '_movie.mp4'))
        print('calling {}'.format(cmd))
        os.system(cmd)

def rolling_average(a, n=5) :
    if n == 0:
        return a
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_states(last_steps, load_model_base, detail_dict):
    st, ac, re, nst, nd, fr, nfr = last_steps
    plt.figure()
    plt.title('reward')
    plt.plot(re)
    plt.savefig(load_model_base+'_reward.png')
    plt.xlabel('steps')
    plt.ylabel('step reward')
    plt.close()
   
    for key, indexer in detail_dict.items():
        plt.plot(st[:,indexer])
        plt.figure()
        plt.title(key)
        plt.plot(st[:,indexer])
        plt.savefig(load_model_base+'_state_%s.png'%(key))
        plt.xlabel('steps')
        plt.ylabel(key)
        plt.close()

def plot_position_actions(last_steps, load_model_base, relative=True):
    # only works for jaco
    st, actions, re, nst, nd, fr, nfr = last_steps
    joint_states = st[:,3:3+13]
    joint_next_states = nst[:,3:3+13]
    for an in range(actions.shape[1]):
        plt.figure()
        aname = 'action_%02d'%an
        plt.title(aname)
        if relative:
            plt.plot(actions[:,an], label='cmd rel', lw=3)
            cmd_action = actions[:,an]+joint_states[:,an]
        else:
            cmd_action = actions[:,an]
        plt.plot(cmd_action, label='cmd', lw=2.5)
        plt.plot(joint_next_states[:,an], label='next state', lw=2)
        error = joint_next_states[:,an] - cmd_action
        plt.plot(error, label='pos error', lw=1.5)
        plt.legend()
        plt.savefig(load_model_base+'_action_%02d.png'%(an))
        plt.xlabel('steps')
        plt.close()

def plot_replay_reward(replay_buffer, load_model_base, start_step=0, name_modifier=''):
    st = np.array(replay_buffer.episode_start_times)
    plt.figure()
    plt.title("Episode Time")
    plt.plot(st[1:]-st[:-1])
    plt.savefig(load_model_base+'_seconds_episode_%s.png'%name_modifier)
    plt.xlabel('episode')
    plt.ylabel('seconds')
    plt.close()

    plt.figure()
    plt.title("filtered reward")
    plt.plot(rolling_average(replay_buffer.episode_rewards, 10))
    plt.savefig(load_model_base+'_rewards_episode_filt_%s.png'%name_modifier)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.close()

    plt.figure()
    plt.title("reward")
    plt.plot(replay_buffer.episode_rewards)
    plt.savefig(load_model_base+'_rewards_episode_%s.png'%name_modifier)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.close()

    plt.figure()
    plt.title("cumulative reward")
    plt.plot(np.cumsum(replay_buffer.episode_rewards))
    plt.savefig(load_model_base+'_cumulative_episode_%s.png'%name_modifier)
    plt.xlabel('episode')
    plt.ylabel('total reward')
    plt.close()

    plt.figure()
    plt.title("reward")
    plt.plot(np.array(replay_buffer.episode_start_steps[1:])+start_step, replay_buffer.episode_rewards)
    plt.savefig(load_model_base+'_rewards_step_%s.png'%name_modifier)
    plt.xlabel('steps')
    plt.ylabel('reward')
    plt.close()

    plt.figure()
    plt.title("cumulative reward")
    plt.plot(np.array(replay_buffer.episode_start_steps[1:])+start_step, np.cumsum(replay_buffer.episode_rewards))
    plt.savefig(load_model_base+'_cumulative_step_%s.png'%name_modifier)
    plt.xlabel('steps')
    plt.ylabel('total reward')
    plt.close()
 

