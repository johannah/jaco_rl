# warning - slow on gpu  - jaco torch functions need to be examined to determine why so slow on gpu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys
from glob import glob
from IPython import embed
import pickle
import numpy as np
from collections import OrderedDict
from copy import deepcopy
import torch
from torch import nn, optim
from torchvision import transforms
from torch.nn.utils.clip_grad import clip_grad_value_
from acn_models import tPTPriorNetwork, weights_init, VQEmbedding, ResBlock, ACNstateAngleEmbedding
from acn_utils import kl_loss_function
from acn_utils import tsne_plot
from acn_utils import pca_plot
from sklearn.cluster import KMeans
from jaco import torchDHtransformer, DH_attributes_jaco27DOF,  find_joint_coordinate_extremes
import time
from plotting import plot_rec_losses, plot_rec_results
from jaco import torchDHtransformer, DH_attributes_jaco27DOF, get_torch_attributes, find_joint_coordinate_extremes
from plotting import plot_rec_losses, plot_rec_results
torch_dh = torchDHtransformer(DH_attributes_jaco27DOF, 'cpu')
tdh = get_torch_attributes(DH_attributes_jaco27DOF, 'cpu')
dh = DH_attributes_jaco27DOF

def batch_torch_dh_transform(dh_index,angles):
    theta = tdh['DH_theta_sign'][dh_index]*angles+tdh['DH_theta_offset'][dh_index]
    d = tdh['DH_d'][dh_index]
    a = tdh['DH_a'][dh_index]
    alpha = tdh['DH_alpha'][dh_index]
    bs = angles.shape[0]
    T = torch.zeros((bs,4,4), device=args.device)
    T[:,0,0] = T[:,0,0] +  torch.cos(theta)
    T[:,0,1] = T[:,0,1] + -torch.sin(theta)*torch.cos(alpha)
    T[:,0,2] = T[:,0,2] +  torch.sin(theta)*torch.sin(alpha)
    T[:,0,3] = T[:,0,3] +  a*torch.cos(theta)
    T[:,1,0] = T[:,1,0] +  torch.sin(theta)
    T[:,1,1] = T[:,1,1] +   torch.cos(theta)*torch.cos(alpha)
    T[:,1,2] = T[:,1,2] +   -torch.cos(theta)*torch.sin(alpha)
    T[:,1,3] = T[:,1,3] +  a*torch.sin(theta)
    #T:,[2,0] =  :,0.0
    T[:,2,1] = T[:,2,1] +  torch.sin(alpha)
    T[:,2,2] = T[:,2,2] +   torch.cos(alpha)
    T[:,2,3] = T[:,2,3] +  d
    #T[3,0] = T[3,0] +  0.0
    #T[3,1] = T[3,1] +  0.0
    #T[3,2] = T[3,2] +  0.0
    T[:,3,3] = T[:,3,3] +  1.0
    ###- .0002 cpu, .005 gpu
    return T 


def forward_pass(state_bins, extremes, Ts, phase, batch_indexes):
    assert state_bins[:,6].min() > 29999
    assert state_bins[:,1].max() < 10001
    #nans out of the model
    u_q = model(state_bins)
    u_q_flat = u_q.contiguous()
    rec_angle = np.pi*(torch.tanh(model.acn_decode(u_q))+1)
    if phase == 'train':
        assert batch_indexes.max() < prior_model.codes.shape[0]
        prior_model.update_codebook(batch_indexes, u_q_flat.detach())
    u_p, s_p = prior_model(u_q_flat)
    Tall = torch.zeros((state_bins.shape[0],4,4)).to(args.device)
    Tall = Tall + torch.FloatTensor(([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])).to(args.device)

    _T0 = batch_torch_dh_transform(0, rec_angle[:,0])
    T0_pred = torch.matmul(Tall,_T0)

    _T1 = batch_torch_dh_transform(1, rec_angle[:,1])
    T1_pred = torch.matmul(T0_pred,_T1)

    _T2 = batch_torch_dh_transform(2, rec_angle[:,2])
    T2_pred = torch.matmul(T1_pred,_T2)

    _T3 = batch_torch_dh_transform(3, rec_angle[:,3])
    T3_pred = torch.matmul(T2_pred,_T3)

    _T4 = batch_torch_dh_transform(4, rec_angle[:,4])
    T4_pred = torch.matmul(T3_pred,_T4)

    _T5 = batch_torch_dh_transform(5, rec_angle[:,5])
    T5_pred = torch.matmul(T4_pred,_T5)

    _T6 = batch_torch_dh_transform(6, rec_angle[:,6])
    T6_pred = torch.matmul(T5_pred,_T6)

    ext_pred = T6_pred[:,:3,3]
    if torch.isnan(u_q).sum():
        print("NAANS")
        embed()
   
    return ext_pred, rec_angle, u_q, u_p, s_p

def run(train_cnt):
    log_ones = torch.zeros(batch_size, args.code_length).to(device)
    st_tm = time.time()
    for dataset_view in range(10000):
        print('starting epoch {}'.format(dataset_view))
        print('last epoch took {:.2f} mins'.format((time.time()-st_tm)/60.0))
        st_tm = time.time()
        for phase in ['valid', 'train']:
            indexes = np.arange(0, data_buffer[phase]['states'].shape[0])
            random_state.shuffle(indexes)
            rec_accum = 0
            kl_accum = 0
            for st in np.arange(0, indexes.shape[0], batch_size):
                en = min([st+batch_size, indexes.shape[0]])
                batch_indexes = indexes[st:en]
                if log_ones.shape[0] != batch_indexes.shape[0]:
                    log_ones = torch.zeros(batch_indexes.shape[0], args.code_length).to(device)
                states = torch.FloatTensor(data_buffer[phase]['states'][batch_indexes]).to(device)
                state_bins = torch.LongTensor(data_buffer[phase]['state_bins'][batch_indexes]).to(device)
                extremes = torch.FloatTensor(data_buffer[phase]['extremes'][batch_indexes,6]).to(device)
                Ts = torch.FloatTensor(data_buffer[phase]['Ts'][batch_indexes]).to(device)
                ext_pred, rec_angle, u_q, u_p, s_p = forward_pass(state_bins, extremes, Ts, phase, batch_indexes)
                rec_loss = mse_loss(extremes, ext_pred.to(device))
                rec_loss = rec_loss*args.rec_weight
                rec_angle.retain_grad()
                kl = kl_loss_function(u_q, 
                                      log_ones,
                                      u_p, 
                                      s_p,
                                      reduction='mean')
 
                if not st %(batch_size*100):
                    print(rec_loss.item(), kl.item())
                #vq_loss = mse_loss(z_q_x, z_e_x.detach())
                #commit_loss = mse_loss(z_e_x, z_q_x.detach())*vq_commitment_beta
                if phase == 'train':
                    clip_grad_value_(model.parameters(), 5)
                    clip_grad_value_(prior_model.parameters(), 5)
                    kl.backward(retain_graph=True)
                    rec_loss.backward() 
                    opt.step()
                    train_cnt += states.shape[0] 

                if st == 0:
                    losses[phase]['rec'].append(rec_loss.detach().cpu().item())
                    losses[phase]['kl'].append(kl.detach().cpu().item())
                    losses[phase]['steps'].append(train_cnt)

                    if phase == 'train' and not dataset_view%args.save_every_epoch:
                        model_dict = {'model':model.state_dict(), 
                                      'prior_model':prior_model.state_dict(), 
                                      'train_cnt':train_cnt, 
                                      'losses':losses
                                    }
                        mpath = os.path.join(savebase, 'model_%012d.pt'%train_cnt)
                        lpath = os.path.join(savebase, 'model_%012d.losses'%train_cnt)
                        torch.save(model_dict, mpath)
                        torch.save(losses, lpath)
                        print('saving {}'.format(mpath))

def create_latent_file(phase, out_path):
    model.eval()
    prior_model.eval()
    if 1:
    #if not os.path.exists(out_path+'.npz'):
        sz = min([data_buffer[phase]['states'].shape[0], 10000])
        all_indexes = []; all_st = []
        all_rec_st = []
        all_acn_uq = []; all_acn_sp = []
        all_neighbors = []; all_neighbor_indexes = []
        all_neighbor_distances = []; all_vq_latents = []
        all_rec_extremes = []; all_extremes = []; all_neighbor_extremes = []
        frames = []; next_frames = []; all_diffs = []

        train_states = deepcopy(data_buffer['train']['states'])
        train_extremes = deepcopy(data_buffer['train']['extremes'])[:,6]
        for i in range(0,sz,batch_size):
            batch_indexes = np.arange(i, min([i+batch_size, sz]))
            states = torch.FloatTensor(data_buffer[phase]['states'][batch_indexes]).to(args.device)
            state_bins = torch.LongTensor(data_buffer[phase]['state_bins'][batch_indexes]).to(device)
            extremes = torch.FloatTensor(data_buffer[phase]['extremes'][batch_indexes,6]).to(device)
            Ts = torch.FloatTensor(data_buffer[phase]['Ts'][batch_indexes]).to(device)
            ext_pred, rec_angle, u_q, u_p, s_p = forward_pass(state_bins, extremes, Ts, phase, batch_indexes)
            states = deepcopy(states.detach().cpu().numpy())
            np_rec_states = deepcopy(rec_angle.detach().cpu().numpy())
            np_rec_extremes = ext_pred.detach().cpu().numpy()
            neighbor_distances, neighbor_indexes = prior_model.kneighbors(u_q.contiguous(), n_neighbors=num_k)
            extremes = data_buffer[phase]['extremes'][batch_indexes][:,6]
            diff = np.sqrt(((extremes-np_rec_extremes)**2).sum(axis=1))
            all_diffs.extend(diff)
            ni = deepcopy(neighbor_indexes.detach().cpu().numpy())
            all_st.extend(states)
            all_rec_st.extend(np_rec_states)
            all_extremes.extend(extremes)
            all_rec_extremes.extend(np_rec_extremes)
            all_acn_uq.extend(deepcopy(u_q.detach().cpu().numpy()))
            all_acn_sp.extend(deepcopy(s_p.detach().cpu().numpy()))
            all_neighbor_indexes.extend(ni)
            all_neighbor_distances.extend(deepcopy(neighbor_distances.detach().cpu().numpy()))
            #all_vq_latents.extend(latents.detach().cpu().numpy())
        print("creating %s latent file: %s with %s examples"%(phase, out_path+'.npz', len(all_st)))
        np.savez(out_path,
                 index=all_indexes,
                 st=all_st,
                 rec_st=all_rec_st,
                 ex=all_extremes,
                 rec_ex=all_rec_extremes,
                 acn_uq=all_acn_uq,
                 acn_sp=all_acn_sp,
                 neighbor_train_indexes=all_neighbor_indexes,
                 neighbor_distances=all_neighbor_distances,
                 train_st=train_states, 
                 train_ex=train_extremes,
                 num_k=num_k,
                 diffs=all_diffs
                ) 


if __name__ == '__main__':
    # trained with 32, nan with 8 codelength
    import argparse
    from datetime import date
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', default=False, action='store_true')
    parser.add_argument('--load_model', default='')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--savedir', default='models')
    parser.add_argument('--exp_name', default='test')
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--plot_random', default=False, action='store_true')
    parser.add_argument('--num_plot_examples', type=int, default=512)
    parser.add_argument('--save_every_epoch', type=int, default=5)
    parser.add_argument('--code_length', type=int, default=8)
    parser.add_argument('--rec_weight', type=float, default=10000)
    cluster_img_size = (120,120) 
    random_state = np.random.RandomState(0)
    today = date.today()
    today_str = today.strftime("%y-%m-%d")
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit()

    if args.load_model == '':
        cnt = 0
        exp_desc = 'acn_embed_%02dCL_%05dwt'%(args.code_length, args.rec_weight)
      
        savebase = os.path.join(args.savedir, today_str+exp_desc+args.exp_name+'_%02d'%cnt)
        while len(glob(os.path.join(savebase, '*.pt'))):
            cnt += 1
            savebase = os.path.join(args.savedir, today_str+exp_desc+args.exp_name+'_%02d'%cnt)
        if not os.path.exists(savebase):
            os.makedirs(savebase)
        if not os.path.exists(os.path.join(savebase, 'python')):
            os.makedirs(os.path.join(savebase, 'python'))
        os.system('cp *.py %s/python'%savebase)
    else:
        savebase = os.path.split(args.load_model)[0]
    print('savebase: {}'.format(savebase))



    valid_fname = 'datasets/test_TD3_jaco_00000_0001140000_eval_CAM-1_S_S1140000_E0010_norm_position_norm_Q005000.npz'
    train_fname = 'datasets/sep8randomClose_TD3_jaco_00000_0000200000_norm_position_norm_Q005000.npz'
    #train_fname = 'datasets/test_TD3_jaco_00000_0001160000_eval_CAM-1_S_S1160000_E0100_norm_position_norm.npz'
    data_buffer = {}
    data_buffer['train'] = np.load(train_fname)
    data_buffer['valid'] = np.load(valid_fname)
    batch_size = 128
    #vq_commitment_beta = 0.25
    num_k = 5
    device = args.device
    torch_dh = torchDHtransformer(DH_attributes_jaco27DOF, device)

    #losses = {
    #          'train':{'kl':[],'rec':[],'vq':[],'commit':[],'steps':[]},
    #          'valid':{'kl':[],'rec':[],'vq':[],'commit':[],'steps':[]}
    #          }
    losses = {
              'train':{'kl':[],'rec':[],'steps':[]},
              'valid':{'kl':[],'rec':[],'steps':[]}
              }
 
    train_size,state_size = data_buffer['train']['states'].shape
    valid_size = data_buffer['valid']['states'].shape[0]
    n_bins = 5000
    n_joints = 7
    model = ACNstateAngleEmbedding(n_bins=n_bins*n_joints, code_length=args.code_length, input_size=state_size).to(device)
    print("setting up prior with training size {}".format(train_size))
    prior_model = tPTPriorNetwork(size_training_set=train_size, 
                                  code_length=args.code_length,
                                  k=num_k).to(device)

    train_cnt = 0
    if args.load_model != '':
        if '.pt' in args.load_model:
            load_path = args.load_model
        else:
            assert os.path.isdir(args.load_model)
            from glob import glob 
            search = os.path.join(args.load_model, '*.pt')
            print('searching {} for models'.format(search))
            found_models = sorted(glob(search))
            print('found {} models'.format(len(found_models)))
            load_path = found_models[-1]
        print('loading {}'.format(load_path))

        results_base = load_path.replace('.pt', '_latent_')
        model_dict = torch.load(load_path)
        losses = torch.load(load_path.replace('.pt', '.losses'))
        model.load_state_dict(model_dict['model'])
        prior_model.load_state_dict(model_dict['prior_model'])
        train_cnt = model_dict['train_cnt']
    if args.plot:
        plot_rec_losses(load_path, losses)
        for phase in ['valid', 'train']:
            create_latent_file(phase, results_base+phase)
        print('plot latents')
        plot_rec_results(results_base+'train.npz', 'train', random_indexes=args.plot_random, num_plot_examples=args.num_plot_examples, plot_neighbors=num_k)
        plot_rec_results(results_base+'valid.npz', 'valid', random_indexes=args.plot_random, num_plot_examples=args.num_plot_examples, plot_neighbors=num_k)
    else:
        mse_loss = nn.MSELoss(reduction='mean')
        parameters = []
        parameters+=list(model.parameters())
        parameters+=list(prior_model.parameters())
        opt = optim.Adam(parameters, lr=1e-6)
        run(train_cnt)

