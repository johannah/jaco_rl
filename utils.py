import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

def plot_losses():
    plt_base = load_path.replace('.pt', '')
    plt.figure()
    for lparam in losses['train'].keys():
        if lparam != 'steps':
            plt.plot(losses['train']['steps'][1:], losses['train'][lparam][1:], label='tr %s'%lparam, marker='x')
            plt.plot(losses['valid']['steps'][1:], losses['valid'][lparam][1:], label='va %s'%lparam, marker='o')
    plt.title('losses')
    plt.legend()
    plt.savefig(plt_base+'_losses.png')
    plt.close()

def plot_results(results_fpath, phase):
    data = np.load(results_fpath)
    ymin = min([data['st'].min(), data['rec_st'].min()])
    ymax = min([data['st'].max(), data['rec_st'].max()])
    exmin = min([data['ex'][:,0].min(), data['rec_ex'][:,0].min()])
    exmax = max([data['ex'][:,0].max(), data['rec_ex'][:,0].max()])
    eymin = min([data['ex'][:,1].min(), data['rec_ex'][:,1].min()])
    eymax = max([data['ex'][:,1].max(), data['rec_ex'][:,1].max()])
    embed()
 
    images = []
    if args.plot_random:
        inds = random_state.randint(0, data['st'].shape[0], args.num_plot_examples).astype(np.int)
    
        pltdir = load_path.replace('.pt', '_plots_random_%s'%phase)
    else:
        inds = np.arange(0, min([args.num_plot_examples,data['st'].shape[0]])).astype(np.int)
        pltdir = load_path.replace('.pt', '_plots_time_%s'%phase)
    print('plotting to', pltdir)

    if not os.path.exists(pltdir):
        os.makedirs(pltdir)
    sscale = 50

    plt.figure()
    plt.plot(data['diffs'])
    plt.savefig(load_path.replace('.pt', '_'+phase+'_TPdiff.png'))
    plt.close()


    for i in inds:
        f,ax = plt.subplots(1,2,figsize=(8,5))
        ex = data['ex'][i]
        rec_ex = data['rec_ex'][i]
 
        ax[0].plot(data['st'][i], label='data', lw=2, marker='x', color='mediumorchid')
        ax[0].plot(data['rec_st'][i], label='rec', lw=1.5, marker='o', color='blue')

        ax[1].scatter(ex[0], ex[1], s=ex[2]*sscale, marker='o', color='mediumorchid')
        ax[1].scatter(rec_ex[0], rec_ex[1], s=rec_ex[2]*sscale, marker='o', color='blue')
        ax[0].set_ylim(ymin, ymax)                    
        ax[1].set_ylim(-.3, 1.3)                    
        ax[1].set_xlim(-.3, .3)                    
        d = data['diffs'][i]
        f.suptitle('TP:[{:.2f},{:.2f},{:.2f}]\nETP:[{:.2f},{:.2f},{:.2f}]\ndiff:[{:.2f},{:.2f},{:.2f}]'.format(ex[0],ex[1],ex[2],rec_ex[0],rec_ex[1],rec_ex[2],d[0],d[1],d[2],))
 
        #plt.legend()
        pltname = os.path.join(pltdir, '%s_%05d.png'%(phase,i))
        ax[0].legend()
        print('plotting %s'%pltname)
        plt.savefig(pltname)
        plt.close()
    
    pltsearch = os.path.join(pltdir, phase+'_*png')
    cmd = 'convert %s %s/_out.gif'%(pltsearch, pltdir)
    os.system(cmd)



# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477
def seed_everything(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class BasePolicy():
    def __init__(
        self,
        state_dim,
        action_dim,
        min_action,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        seed=1222,
        device='cpu'
    ):
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.max_action = max_action
        self.min_action = min_action
        self.action_dim = action_dim
        self.state_dim = state_dim

    def select_action(self, state):
        action = np.random.uniform(low=self.min_action, high=self.max_action, size=self.action_dim)
        return action

    def train(self, replay_buffer, batch_size=100):
        pass

    def save(self, filename):
        pass

    def load(self, filename):
        pass


class RandomPolicy(BasePolicy):
    def select_action(self, state):
        action = np.random.uniform(low=self.min_action, high=self.max_action, size=self.action_dim)
        return action

def save_info_dict(info, base_save_path):
    pickle.dump(info, open(base_save_path+'.infopkl', 'wb'))

def load_info_dict(base_load_path):
    return pickle.load(open(base_load_path+'.infopkl', 'rb'))

def create_new_info_dict(arg_dict, load_model_base='', load_replay_path=''):
    info = {
            'load_model_path':load_model_base+'.pt',
            'load_replay_path':load_replay_path,
            'save_start_times':[],
            'save_end_times':[],
            'args':[arg_dict],
             }
    return info
