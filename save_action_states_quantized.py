import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy
import pickle
from replay_buffer import ReplayBuffer, compress_frame
from TD3 import TD3
import torch
from jaco import DHtransform, find_joint_coordinate_extremes,  torchDHtransformer, DH_attributes_jaco27DOF
import numpy as np
import os
import sys

from IPython import embed
torch_dh = torchDHtransformer(DH_attributes_jaco27DOF, 'cpu')
kwargs = {'state_dim':16, 'action_dim':7, 'max_action':.1}
policy = TD3(**kwargs)

load_dir = 'results/sep8randomClose02/'

va_eval_dir = 'sep8randomClose_TD3_jaco_00000_0001140000_eval_CAM-1_S_S1140000_E0010/'
va_sim_replay_file = 'test_TD3_jaco_00000_0001140000_eval_CAM-1_S_S1140000_E0010.epkl'


#tr_eval_dir = 'sep8randomClose_TD3_jaco_00000_0001160000_eval_CAM-1_S_S1160000_E0100'
#tr_sim_replay_file = 'test_TD3_jaco_00000_0001160000_eval_CAM-1_S_S1160000_E0100.epkl'
tr_eval_dir = ''
#tr_sim_replay_file = 'sep8randomClose_TD3_jaco_00000_0000600000.pkl'
tr_sim_replay_file = 'sep8randomClose_TD3_jaco_00000_0000200000.pkl'
tr_path = os.path.join(load_dir, tr_eval_dir, tr_sim_replay_file)
va_path = os.path.join(load_dir, va_eval_dir, va_sim_replay_file)
 
n_bins = 5000
for fp in  [va_path, tr_path]:

    fn = os.path.split(fp)[1]
    if '.epkl' in fn:
        savename = 'datasets/{}_position_norm_Q{:06d}'.format(fn.replace('.epkl', '_norm'), n_bins)
    elif '.pkl' in fn:
        savename = 'datasets/{}_position_norm_Q{:06d}'.format(fn.replace('.pkl', '_norm'), n_bins)
    else:
        raise
    
    imgsavename = savename+'_img'

    sbuffer = pickle.load(open(fp, 'rb'))
    
    states = sbuffer.states[:sbuffer.size,3:3+7] 
    next_states = sbuffer.next_states[:sbuffer.size,3:3+7] 
    actions = sbuffer.actions[:sbuffer.size] 
    #frames = np.array([sbuffer.undo_frame_compression(sbuffer.frames[x]) for x in range(states.shape[0])])
    #next_frames = np.array([sbuffer.undo_frame_compression(sbuffer.next_frames[x]) for x in range(states.shape[0])])
    extremes = []
    Ts = []
    Es = []
    S_norms = []
    st_norms = deepcopy(states)
    st_norms[st_norms<0] = st_norms[st_norms<0]+(2*np.pi)
    st_norms = st_norms%(2*np.pi)
    
    dTs = []
    dEs = []
    dst_norms = []
    bins = np.linspace(0,2*np.pi, n_bins)
    st_bins = np.digitize(st_norms, bins)
    dst_norms = bins[st_bins]
    for jt in range(st_bins.shape[1]):
        st_bins[:,jt] += n_bins*jt
 
    for i in range(states.shape[0]):
        # unnormalized version
        ex_norm, T_norm = torch_dh.find_joint_coordinate_extremes(torch.FloatTensor(st_norms[i]), base_tTall=torch_dh.base_tTall, return_T=True)
        ex_norm_np = ex_norm.detach().numpy()
        T_norm_np = [t.detach().numpy() for t in T_norm]
        Ts.append(T_norm_np)
        Es.append(ex_norm_np)
    for i in range(states.shape[0]):
        # unnormalized version
        dex_norm, dT_norm = torch_dh.find_joint_coordinate_extremes(torch.FloatTensor(dst_norms[i]), base_tTall=torch_dh.base_tTall, return_T=True)
        dex_norm_np = dex_norm.detach().numpy()
        dT_norm_np = [t.detach().numpy() for t in dT_norm]
        dTs.append(dT_norm_np)
        dEs.append(dex_norm_np)

    if not os.path.exists(imgsavename):
        os.mkdir(imgsavename)
    for x in range(50):
        f,ax = plt.subplots(1,2)
        ax[0].plot(st_norms[x])
        ax[0].plot(dst_norms[x])
        ax[1].scatter(Es[x][-1][0], Es[x][-1][1])
        ax[1].scatter(dEs[x][-1][0], dEs[x][-1][1])
        ax[0].set_ylim(0, 2*np.pi)
        ax[1].set_ylim(-.4, .4)
        ax[1].set_xlim(-.4, .4)
        plt.title('{}:\nE{}\n dE{}'.format(x, Es[x][-1], dEs[x][-1]))
        plt.savefig('%s/img%05d.png'%(imgsavename,x))
        plt.close()
    cmd = 'ffmpeg -y -i %s'%(imgsavename)  + '/img%05d.png'+ ' %s/_output.mp4'%(imgsavename)
    os.system(cmd)

    for i in range(states.shape[0]):
        # unnormalized version
        
        ex_norm, T_norm = torch_dh.find_joint_coordinate_extremes(torch.FloatTensor(st_norms[i]), base_tTall=torch_dh.base_tTall, return_T=True)
        ex_norm_np = ex_norm.detach().numpy()
        T_norm_np = [t.detach().numpy() for t in T_norm]
        Ts.append(T_norm_np)
        Es.append(ex_norm_np)
 
    print('saving', savename)
    np.savez(savename, states=st_norms, dstates=dst_norms, bins=bins, state_bins=st_bins, actions=actions, extremes=Es, dextremes=dEs,  Ts=Ts, dTs=dTs)
    #np.savez(savename, states=states, next_states=next_states,  actions=actions, extremes=Es, Ts=Ts, cam_dim=sbuffer.cam_dim)
    

