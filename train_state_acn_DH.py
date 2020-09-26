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
from acn_models import tPTPriorNetwork, weights_init, VQEmbedding, ResBlock, ACNstateAngle
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

def torch_dh_transform(dh_index,angle):
    theta = tdh['DH_theta_sign'][dh_index]*angle+tdh['DH_theta_offset'][dh_index]
    d = tdh['DH_d'][dh_index]
    a = tdh['DH_a'][dh_index]
    alpha = tdh['DH_alpha'][dh_index]
    T = torch.zeros((4,4), device='cpu')
    T[0,0] = T[0,0] +  torch.cos(theta)
    T[0,1] = T[0,1] + -torch.sin(theta)*torch.cos(alpha)
    T[0,2] = T[0,2] +  torch.sin(theta)*torch.sin(alpha)
    T[0,3] = T[0,3] +  a*torch.cos(theta)
    T[1,0] = T[1,0] +  torch.sin(theta)
    T[1,1] = T[1,1] +   torch.cos(theta)*torch.cos(alpha)
    T[1,2] = T[1,2] +   -torch.cos(theta)*torch.sin(alpha)
    T[1,3] = T[1,3] +  a*torch.sin(theta)
    #T[2,0] =  0.0
    T[2,1] = T[2,1] +  torch.sin(alpha)
    T[2,2] = T[2,2] +   torch.cos(alpha)
    T[2,3] = T[2,3] +  d
    #T[3,0] = T[3,0] +  0.0
    #T[3,1] = T[3,1] +  0.0
    #T[3,2] = T[3,2] +  0.0
    T[3,3] = T[3,3] +  1.0
    ###- .0002 cpu, .005 gpu
    return T 


def run(train_cnt):
    log_ones = torch.zeros(batch_size, code_length).to(device)
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
                    log_ones = torch.zeros(batch_indexes.shape[0], code_length).to(device)
                states = torch.FloatTensor(data_buffer[phase]['states'][batch_indexes]).to(device)
                #extremes = torch.FloatTensor(data_buffer[phase]['extremes'][batch_indexes]).to(device)
                
                u_q = model(states)
                u_q_flat = u_q.contiguous()
                if phase == 'train':
                    assert batch_indexes.max() < prior_model.codes.shape[0]
                    prior_model.update_codebook(batch_indexes, u_q_flat.detach())
                u_p, s_p = prior_model(u_q_flat)
                extremes = torch.FloatTensor(data_buffer[phase]['extremes'][batch_indexes,6]).to(device)
                # predict angle 6
                rec_states = np.pi*(torch.tanh(model.acn_decode(u_q))+1.0)


                if args.use_DH:
                    # do dh on cpu
                    
                    Tbase = torch.FloatTensor(data_buffer[phase]['Ts'][batch_indexes,0]).to('cpu')
                    ext_pred = torch.zeros((batch_indexes.shape[0], 3)).to('cpu')
                    for x in range(rec_states.shape[0]):
                        T0_pred = torch.matmul(Tbase[x], torch_dh_transform(0, rec_states[x,0]))
                        T1_pred = torch.matmul(T0_pred, torch_dh_transform(1, rec_states[x,1]))
                        T2_pred = torch.matmul(T1_pred, torch_dh_transform(2, rec_states[x,2]))
                        T3_pred = torch.matmul(T2_pred, torch_dh_transform(3, rec_states[x,3]))
                        T4_pred = torch.matmul(T3_pred, torch_dh_transform(4, rec_states[x,4]))
                        T5_pred = torch.matmul(T4_pred, torch_dh_transform(5, rec_states[x,6]))
                        T6_pred = torch.matmul(T5_pred, torch_dh_transform(6, states[x,6]))

                        ext_pred[x] = ext_pred[x] + T6_pred[:3,3]

                    # TODO fix X
                    rec_loss = mse_loss(extremes, ext_pred.to(device))
                else:
                    rec_loss = mse_loss(states, rec_states)
                rec_loss = rec_loss*(rec_states.shape[1]*args.rec_weight)
                rec_states.retain_grad()
                kl = kl_loss_function(u_q, 
                                      log_ones,
                                      u_p, 
                                      s_p,
                                      reduction='mean')
 
                rec_accum += rec_loss.detach().cpu().item()
                kl_accum += kl.detach().cpu().item()
                #vq_loss = mse_loss(z_q_x, z_e_x.detach())
                #commit_loss = mse_loss(z_e_x, z_q_x.detach())*vq_commitment_beta
                if phase == 'train':
                    clip_grad_value_(model.parameters(), 10)
                    clip_grad_value_(prior_model.parameters(), 10)
                    kl.backward(retain_graph=True)
                    rec_loss.backward() 
                    opt.step()
                    train_cnt += states.shape[0] 

                if st == 0:
                    losses[phase]['kl'].append(rec_accum/indexes.shape[0])
                    losses[phase]['rec'].append(kl_accum/indexes.shape[0])
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
    if not os.path.exists(out_path+'.npz'):
        sz = data_buffer[phase]['states'].shape[0]
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
            u_q = model(states)
            u_q_flat = u_q.contiguous()
            u_p, s_p = prior_model(u_q_flat)
            rec_states = np.pi*(torch.tanh(model.acn_decode(u_q))+1.0)
            states = deepcopy(states.detach().cpu().numpy())
            np_rec_states = deepcopy(rec_states.detach().cpu().numpy())
            if args.use_DH:
                Tbase = torch.FloatTensor(data_buffer[phase]['Ts'][batch_indexes,0]).to('cpu')
                ext_pred = torch.zeros((batch_indexes.shape[0], 3)).to('cpu')
                for x in range(rec_states.shape[0]):
                    T0_pred = torch.matmul(Tbase[x], torch_dh_transform(0, rec_states[x,0]))
                    T1_pred = torch.matmul(T0_pred, torch_dh_transform(1, rec_states[x,1]))
                    T2_pred = torch.matmul(T1_pred, torch_dh_transform(2, rec_states[x,2]))
                    T3_pred = torch.matmul(T2_pred, torch_dh_transform(3, rec_states[x,3]))
                    T4_pred = torch.matmul(T3_pred, torch_dh_transform(4, rec_states[x,4]))
                    T5_pred = torch.matmul(T4_pred, torch_dh_transform(5, rec_states[x,6]))
                    T6_pred = torch.matmul(T5_pred, torch_dh_transform(6, states[x,6]))
                    ext_pred[x] = ext_pred[x] + T6_pred[:3,3]

            #np_rec_extremes = np.array([find_joint_coordinate_extremes(DH_attributes_jaco27DOF, np_rec_states[x]) for x in range(states.shape[0])])[:,6]
            np_rec_extremes = ext_pred.detach().cpu().numpy()
 
            neighbor_distances, neighbor_indexes = prior_model.kneighbors(u_q_flat, n_neighbors=num_k)

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
    import argparse
    from datetime import date
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', default=False, action='store_true')
    parser.add_argument('--use_DH', default=False, action='store_true')
    parser.add_argument('--load_model', default='')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--savedir', default='models')
    parser.add_argument('--exp_name', default='test')
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--plot_random', default=False, action='store_true')
    parser.add_argument('--num_plot_examples', default=256)
    parser.add_argument('--save_every_epoch', default=1)
    parser.add_argument('--rec_weight', type=float, default=10)
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
        exp_desc = 'acn_dhv110wt_norm' 
        if args.use_DH:
            exp_desc = exp_desc+"DH"
      
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



    valid_fname = 'datasets/test_TD3_jaco_00000_0001140000_eval_CAM-1_S_S1140000_E0010_position_norm.npz'
    train_fname = 'datasets/sep8randomClose_TD3_jaco_00000_0000600000_position_norm.npz'
    data_buffer = {}
    data_buffer['train'] = np.load(train_fname)
    data_buffer['valid'] = np.load(valid_fname)
    batch_size = 128
    code_length = 32
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
    model = ACNstateAngle(code_length=code_length, input_size=state_size).to(device)
    print("setting up prior with training size {}".format(train_size))
    prior_model = tPTPriorNetwork(size_training_set=train_size, 
                                  code_length=code_length,
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
        opt = optim.Adam(parameters, lr=1e-5)
        run(train_cnt)

