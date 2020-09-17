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
from jaco import torchDHtransformer, DH_attributes_jaco27DOF, get_torch_attributes

def torch_DHtransform(d,theta,a,alpha):
    # need to add DH params individually to get grad thru - creating new float tensor didnt work
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
    for dataset_view in range(10000):
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
                rec_angle = np.pi*torch.tanh(model(states[:,:6], extremes))

                true_angle = states[:,6]
                # do dh on cpu
              
                T4 = torch.FloatTensor(data_buffer[phase]['Ts'][batch_indexes,4]).to('cpu')
                T5 = torch.FloatTensor(data_buffer[phase]['Ts'][batch_indexes,5]).to('cpu')
                T6 = torch.FloatTensor(data_buffer[phase]['Ts'][batch_indexes,6]).to('cpu')
 
                T_pred = torch.zeros((batch_indexes.shape[0], 2, 4, 4)).to('cpu')
                T_true = torch.stack((T5,T6), 1) .to(device)
        
                for x in range(batch_indexes.shape[0]):
                    DH_theta5 = tdh['DH_theta_sign'][5]*rec_angle[x]+tdh['DH_theta_offset'][6]
                    #DH_theta5 = tdh['DH_theta_sign'][6]*states[x,5]+tdh['DH_theta_offset'][6]
                    _T5 = torch_DHtransform(tdh['DH_d'][5], DH_theta5, tdh['DH_a'][5], tdh['DH_alpha'][5])
                    _T5.to('cpu')
 
                    T5_pred = torch.matmul(T4[x],_T5)
                    T_pred[x,0] = T_pred[x,0] + T5_pred

                    DH_theta6 = tdh['DH_theta_sign'][6]*states[x,6]+tdh['DH_theta_offset'][6]
                    #DH_theta6 = tdh['DH_theta_sign'][6]*true_angle[x]+tdh['DH_theta_offset'][6]
                    _T6 = torch_DHtransform(tdh['DH_d'][6], DH_theta6, tdh['DH_a'][6], tdh['DH_alpha'][6])
                    _T6.to('cpu')
 
                    T6_pred = torch.matmul(T5_pred,_T6)
                    T_pred[x,1] = T_pred[x,1] + T6_pred


                # TODO fix X
                rec_loss = mse_loss(T_true, T_pred.to(device))
                #rec_loss = mse_loss(extremes,extreme_ends)
                #rec_loss = mse_loss(states[:,5],rec_angle)
                #rec_angle.retain_grad()
                #T_pred.retain_grad()
                #T5_pred.retain_grad()
                #T6_pred.retain_grad()
                if phase == 'train':
                    clip_grad_value_(model.parameters(), 5)
                    rec_loss.backward() 
                    opt.step()
                    train_cnt += batch_size 

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
        frames = []; next_frames = []
        for i in range(0,sz,batch_size):
            batch_indexes = np.arange(i, min([i+batch_size, sz]))

            states = torch.FloatTensor(data_buffer[phase]['states'][batch_indexes]).to(device)
            extremes = torch.FloatTensor(data_buffer[phase]['extremes'][batch_indexes, 6]).to(device)

            # predict angle 6
            rec_angle = np.pi*torch.tanh(model(states[:,:6], extremes))

            true_angle = states[:,6]
            # do dh on cpu
            
            T4 = torch.FloatTensor(data_buffer[phase]['Ts'][batch_indexes,4]).to('cpu')
            T5 = torch.FloatTensor(data_buffer[phase]['Ts'][batch_indexes,5]).to('cpu')
            T6 = torch.FloatTensor(data_buffer[phase]['Ts'][batch_indexes,6]).to('cpu')
 
            T_pred = torch.zeros((batch_indexes.shape[0], 2, 4, 4)).to('cpu')
            T_true = torch.stack((T5,T6), 1) .to(device)
        
            for x in range(batch_indexes.shape[0]):
                DH_theta5 = tdh['DH_theta_sign'][5]*rec_angle[x]+tdh['DH_theta_offset'][6]
                #DH_theta5 = tdh['DH_theta_sign'][6]*states[x,5]+tdh['DH_theta_offset'][6]
                _T5 = torch_DHtransform(tdh['DH_d'][5], DH_theta5, tdh['DH_a'][5], tdh['DH_alpha'][5])
                _T5.to('cpu')
 
                T5_pred = torch.matmul(T4[x],_T5)
                T_pred[x,0] = T_pred[x,0] + T5_pred

                DH_theta6 = tdh['DH_theta_sign'][6]*states[x,6]+tdh['DH_theta_offset'][6]
                #DH_theta6 = tdh['DH_theta_sign'][6]*true_angle[x]+tdh['DH_theta_offset'][6]
                _T6 = torch_DHtransform(tdh['DH_d'][6], DH_theta6, tdh['DH_a'][6], tdh['DH_alpha'][6])
                _T6.to('cpu')
 
                T6_pred = torch.matmul(T5_pred,_T6)
                T_pred[x,1] = T_pred[x,1] + T6_pred


            states = states.detach().cpu().numpy()
            rec_st = deepcopy(states)
            rec_st[:,5] = rec_angle.detach().cpu().numpy()[:,0]

            all_st.extend(states)
            all_extremes.extend(deepcopy(data_buffer[phase]['extremes'][:,6]))
            all_rec_extremes.extend(deepcopy(T_pred[:,1,:3, 3].detach().cpu().numpy()))
            all_rec_st.extend(rec_st)
        print("creating %s results file: %s with %s examples"%(phase, out_path+'.npz', len(all_st)))
        np.savez(out_path,
                 index=all_indexes,
                 st=all_st,
                 rec_st=all_rec_st,
                 ex=all_extremes,
                 rec_ex=all_rec_extremes,
                ) 


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
    ymin = data['st'].min()
    ymax = data['st'].max()
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
    diffs = []
    for i in inds:
        f,ax = plt.subplots(1,2,figsize=(8,5))
        ex = data['ex'][i]
        rec_ex = data['rec_ex'][i]
        diff = np.square(ex-rec_ex)
        diffs.append(diff)
 
        ax[0].plot(data['st'][i], label='data', lw=2, marker='x', color='mediumorchid')
        ax[0].plot(data['rec_st'][i], label='rec', lw=1.5, marker='o', color='blue')

        ax[1].scatter(ex[0], ex[1], s=ex[2]*sscale, marker='o', color='mediumorchid')
        ax[1].scatter(rec_ex[0], rec_ex[1], s=rec_ex[2]*sscale, marker='o', color='blue')
        ax[0].set_ylim(ymin, ymax)                    
        ax[1].set_ylim(-.3, 1.3)                    
        ax[1].set_xlim(-.3, .3)                    
        f.suptitle('TP:[{:.2f},{:.2f},{:.2f}]\nETP:[{:.2f},{:.2f},{:.2f}]\ndiff:[{:.2f},{:.2f},{:.2f}]'.format(ex[0],ex[1],ex[2],rec_ex[0],rec_ex[1],rec_ex[2],diff[0],diff[1],diff[2],))
        #plt.legend()
        pltname = os.path.join(pltdir, '%s_%05d.png'%(phase,i))
        ax[0].legend()
        print('plotting %s'%pltname)
        plt.savefig(pltname)
        plt.close()
    
        images.append(plt.imread(pltname))
        plt.figure()
        plt.plot(diffs)
        plt.savefig(load_path.replace('.pt', phase+'_TPdiff.png'))
        plt.close()
    images = np.array(images)
    pltsearch = os.path.join(pltdir, phase+'_*png')
    cmd = 'convert %s %s/_out.gif'%(pltsearch, pltdir)
    os.system(cmd)

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
    parser.add_argument('--save_every_epochs', default=100)
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
        exp_desc = 'ae5' 
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


    valid_fname = 'datasets/test_TD3_jaco_00000_0001140000_eval_CAM-1_S_S1140000_E0010_position.npz'
    train_fname = 'datasets/test_TD3_jaco_00000_0001160000_eval_CAM-1_S_S1160000_E0100_position.npz'
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
    model = AngleAutoEncoder(input_size=6, output_size=1).to(device)
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
        plot_losses()
        for phase in ['valid', 'train']:
            create_results_file(phase, results_base+phase)
        print('plot results')
        plot_results(results_base+'train.npz', 'train')
        plot_results(results_base+'valid.npz', 'valid')
    else:
        mse_loss = nn.MSELoss(reduction='mean')
        parameters = []
        parameters+=list(model.parameters())
        opt = optim.Adam(parameters, lr=1e-6)
        run(train_cnt)

