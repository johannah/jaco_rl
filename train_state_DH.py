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
from torch.autograd import Variable
from torch import nn, optim
from torchvision import transforms
from torch.nn.utils.clip_grad import clip_grad_value_
from models import AngleAutoEncoder
import time
from plotting import plot_rec_losses, plot_rec_results
from jaco import torchDHtransformer, DH_attributes_jaco27DOF, get_torch_attributes, find_joint_coordinate_extremes

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
    st_tm = time.time()
    for dataset_view in range(train_cnt//batch_size, 10000000):
        print('starting epoch {}'.format(dataset_view))
        print('last epoch took {:.2f} mins'.format((time.time()-st_tm)/60.0))
        st_tm = time.time()
        for phase in ['train', 'valid' ]:
            indexes = np.arange(0, data_buffer[phase]['states'].shape[0])
            random_state.shuffle(indexes)
            for st in np.arange(0, indexes.shape[0], batch_size):
                model.zero_grad()
                en = min([st+batch_size, indexes.shape[0]])
                batch_indexes = indexes[st:en]
                states = torch.FloatTensor(data_buffer[phase]['states'][batch_indexes]).to(device)
                extremes = torch.FloatTensor(data_buffer[phase]['extremes'][batch_indexes,6]).to(device)
                # predict angle 6
                rec_angle = np.pi*(torch.tanh(model(states[:,:3], extremes))+1)

                T2 = torch.FloatTensor(data_buffer[phase]['Ts'][batch_indexes,2]).to('cpu')
                ext_pred = torch.zeros((batch_indexes.shape[0], 3)).to('cpu')
        
                for x in range(batch_indexes.shape[0]):
                    _T3 = torch_dh_transform(3, rec_angle[x,0])
                    T3_pred = torch.matmul(T2[x],_T3)

                    _T4 = torch_dh_transform(4, rec_angle[x,1])
                    T4_pred = torch.matmul(T3_pred,_T4)

                    _T5 = torch_dh_transform(5, rec_angle[x,2])
                    T5_pred = torch.matmul(T4_pred,_T5)

                    _T6 = torch_dh_transform(6, states[x,6])
                    T6_pred = torch.matmul(T5_pred,_T6)

                    ext_pred[x] = ext_pred[x] + T6_pred[:3,3]
 
                rec_loss = mse_loss(extremes, ext_pred.to(device))
                if phase == 'train':
                    clip_grad_value_(model.parameters(), 5)
                    rec_loss.backward() 
                    opt.step()
                    train_cnt += batch_size 

                if st == 0:
                    losses[phase]['rec'].append(rec_loss.detach().cpu().item())
                    losses[phase]['steps'].append(train_cnt)

                    if phase == 'train' and not dataset_view%args.save_every_epochs:
                        model_dict = {'model':model.state_dict(), 
                                      'train_cnt':train_cnt, 
                                      'losses':losses
                                    }
                        mpath = os.path.join(savebase, 'model_%012d.pt'%train_cnt)
                        lpath = os.path.join(savebase, 'model_%012d.losses'%train_cnt)
                        torch.save(model_dict, mpath)
                        torch.save(losses, lpath)
                        print('saving {}'.format(mpath))

def create_results_file(phase, out_path):
    model.eval()
    if not os.path.exists(out_path+'.npz'):
        sz = data_buffer[phase]['states'].shape[0]
        all_indexes = []; all_st = []
        all_rec_st = []
        all_rec_extremes = []; all_extremes = []
        frames = []; next_frames = []; all_diffs = []
        for i in range(0,sz,batch_size):
            batch_indexes = np.arange(i, min([i+batch_size, sz]))
            states = torch.FloatTensor(data_buffer[phase]['states'][batch_indexes]).to(device)
            extremes = torch.FloatTensor(data_buffer[phase]['extremes'][batch_indexes,6]).to(device)

            rec_angle = np.pi*(torch.tanh(model(states[:,:3], extremes))+1)

            T2 = torch.FloatTensor(data_buffer[phase]['Ts'][batch_indexes,2]).to('cpu')
            ext_pred = torch.zeros((batch_indexes.shape[0], 3)).to('cpu')
        
            for x in range(batch_indexes.shape[0]):
                _T3 = torch_dh_transform(3, rec_angle[x,0])
                T3_pred = torch.matmul(T2[x],_T3)

                _T4 = torch_dh_transform(4, rec_angle[x,1])
                T4_pred = torch.matmul(T3_pred,_T4)

                _T5 = torch_dh_transform(5, rec_angle[x,2])
                T5_pred = torch.matmul(T4_pred,_T5)

                _T6 = torch_dh_transform(6, states[x,6])
                T6_pred = torch.matmul(T5_pred,_T6)

                ext_pred[x] = ext_pred[x] + T6_pred[:3,3]
 
            extremes = data_buffer[phase]['extremes'][batch_indexes,6]
            states = states.detach().cpu().numpy()
            rec_st = deepcopy(states)
            rec_st[:,4:6] = rec_angle.detach().cpu().numpy()
            rec_ex = ext_pred.detach().cpu().numpy()

            all_st.extend(states)
            all_extremes.extend(extremes)
            all_rec_st.extend(rec_st)
            all_rec_extremes.extend(rec_ex)
            diff = np.sqrt(((extremes-rec_ex)**2).sum(axis=1))
            all_diffs.extend(diff)
 
        print("creating %s results file: %s with %s examples"%(phase, out_path+'.npz', len(all_st)))
        np.savez(out_path,
                 index=all_indexes,
                 st=all_st,
                 rec_st=all_rec_st,
                 ex=all_extremes,
                 rec_ex=all_rec_extremes,
                 diffs=all_diffs
                ) 



if __name__ == '__main__':
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
    parser.add_argument('--num_plot_examples', default=256)
    parser.add_argument('--save_every_epochs', default=1)
    parser.add_argument('--rec_weight', type=float, default=4000)
    batch_size = 256
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
        exp_desc = 'aepos45normlr1e4' 
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

    device = args.device
    dh = DH_attributes_jaco27DOF
    torch_dh = torchDHtransformer(DH_attributes_jaco27DOF, 'cpu')
    tdh = get_torch_attributes(DH_attributes_jaco27DOF, 'cpu')
    losses = {
              'train':{'rec':[],'steps':[]},
              'valid':{'rec':[],'steps':[]}
              }
 
    train_size,state_size = data_buffer['train']['states'].shape
    valid_size = data_buffer['valid']['states'].shape[0]
    model = AngleAutoEncoder(input_size=3, output_size=3).to(device)
    print("setting up with training size {}".format(train_size))
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

        results_base = load_path.replace('.pt', '_results_')
        model_dict = torch.load(load_path)
        losses = torch.load(load_path.replace('.pt', '.losses'))
        model.load_state_dict(model_dict['model'])
        train_cnt = model_dict['train_cnt']
    if args.plot:
        plot_rec_losses(load_path, losses)
        for phase in ['valid', 'train']:
            create_results_file(phase, results_base+phase)
        print('plot results')
        plot_rec_results(results_base+'train.npz', 'train', random_indexes=args.plot_random, num_plot_examples=args.num_plot_examples)
        plot_rec_results(results_base+'valid.npz', 'valid', random_indexes=args.plot_random, num_plot_examples=args.num_plot_examples)
    else:
        mse_loss = nn.MSELoss(reduction='mean')
        parameters = []
        parameters+=list(model.parameters())
        opt = optim.Adam(parameters, lr=1e-4)
        run(train_cnt)

