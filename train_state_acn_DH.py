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
                #rec_state, z_e_x, z_q_x, latents = model.decode(u_q)
                rec_state = model.acn_decode(u_q)
                if args.use_DH:
                    extremes = torch.FloatTensor(data_buffer[phase]['extremes'][batch_indexes]).to(device)
                    rec_extremes = torch.stack([torch_dh.find_joint_coordinate_extremes(rec_state[x]) for x in range(states.shape[0])]).to(device)
                    rec_loss = mse_loss(extremes[:,-1], rec_extremes[:,-1])*args.rec_weight
                    rec_extremes.retain_grad()
                else:
                    rec_loss = mse_loss(states, rec_state)*(rec_state.shape[-1]*args.rec_weight)
                rec_state.retain_grad()
                kl = kl_loss_function(u_q, 
                                      log_ones,
                                      u_p, 
                                      s_p,
                                      reduction='mean')
 
                #vq_loss = mse_loss(z_q_x, z_e_x.detach())
                #commit_loss = mse_loss(z_e_x, z_q_x.detach())*vq_commitment_beta
                if phase == 'train':
                    clip_grad_value_(model.parameters(), 10)
                    clip_grad_value_(prior_model.parameters(), 10)
                    kl.backward(retain_graph=True)
                    rec_loss.backward() 
                    #vq_loss.backward(retain_graph=True)
                    #commit_loss.backward(retain_graph=True)
                    opt.step()
                    train_cnt += batch_size 
            losses[phase]['kl'].append(kl.detach().cpu().item())
            losses[phase]['rec'].append(rec_loss.detach().cpu().item())
            #losses[phase]['vq'].append(vq_loss.detach().cpu().item())
            #losses[phase]['commit'].append(commit_loss.detach().cpu().item())
            losses[phase]['steps'].append(train_cnt)

            if phase == 'train' and not dataset_view%10:
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
        all_rec_extremes = []; all_extremes = []
        frames = []; next_frames = []
        for i in range(0,sz,batch_size):
            batch_indexes = np.arange(i, min([i+batch_size, sz]))
            states = torch.FloatTensor(data_buffer[phase]['states'][batch_indexes]).to(args.device)
            u_q = model(states)
            u_q_flat = u_q.contiguous()
            u_p, s_p = prior_model(u_q_flat)
            #rec_st, z_e_x, z_q_x, latents = model.decode(u_q)
            rec_states = model.acn_decode(u_q)
            rec_extremes = torch.stack([torch_dh.find_joint_coordinate_extremes(rec_states[x], torch_dh.base_tTall) for x in range(states.shape[0])]).to(device)
            neighbor_distances, neighbor_indexes = prior_model.kneighbors(u_q_flat, n_neighbors=num_k)
            all_st.extend(deepcopy(states.detach().cpu().numpy()))
            all_extremes.extend(deepcopy(data_buffer[phase]['extremes']))
            all_rec_extremes.extend(deepcopy(rec_extremes.detach().cpu().numpy()))
            all_rec_st.extend(deepcopy(rec_states.detach().cpu().numpy()))

            all_acn_uq.extend(deepcopy(u_q.detach().cpu().numpy()))
            all_acn_sp.extend(deepcopy(s_p.detach().cpu().numpy()))
            ni = deepcopy(neighbor_indexes.detach().cpu().numpy())
            all_neighbor_indexes.extend(ni)
            all_neighbors.extend(deepcopy(data_buffer['train']['states'][ni]))
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
                 neighbors=all_neighbors,
                 neighbor_distances=all_neighbor_distances,
                 #vq_latents=all_vq_latents, 
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

def plot_latents(latent_train, latent_valid):
    train_data = np.load(latent_train)
    valid_data = np.load(latent_valid)
    blues = plt.cm.Blues(np.linspace(0.2,.8,num_k))[::-1]
   
    ymin = train_data['st'].min()
    ymax = train_data['st'].max()
    for phase, data in [('valid',valid_data), ('train',train_data)]:
 
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
            ex = data['ex'][i][-1]
            rec_ex = data['rec_ex'][i][-1]
            diff = np.square(ex-rec_ex)
            diffs.append(diff)
 
            ax[0].plot(data['st'][i], label='data', lw=2, marker='x', color='mediumorchid')
            ax[0].plot(data['rec_st'][i], label='rec', lw=1.5, marker='o', color='blue')

            ax[1].scatter(ex[0], ex[1], s=ex[2]*sscale, marker='o', color='mediumorchid')
            ax[1].scatter(rec_ex[0], rec_ex[1], s=rec_ex[2]*sscale, marker='o', color='blue')
             
            for nn, n in enumerate(data['neighbor_train_indexes'][i]):
                if nn in [0, num_k-1]: 
                    ax[0].plot(data['neighbors'][i][nn], label='k:%d i:%d'%(nn,n), lw=1., marker='.', color=blues[nn], alpha=.5)
                else:
                    ax[0].plot(data['neighbors'][i][nn], lw=1., marker='.', color=blues[nn], alpha=.5)
                n_ext = find_joint_coordinate_extremes(DH_attributes_jaco27DOF, data['neighbors'][i][nn])
                ax[1].scatter(n_ext[-1][0], n_ext[-1][1], s=n_ext[-1][2]*sscale, marker='o', color=blues[nn], alpha=0.5)
            ax[0].set_ylim(ymin, ymax)                    
            ax[1].set_ylim(-.2, 1.3)                    
            ax[1].set_xlim(-.25, .25)                    
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
        plt.savefig(load_path.replace('.pt', 'TPdiff.png'))
        images = np.array(images)
        pltsearch = os.path.join(pltdir, phase+'_*png')
        cmd = 'convert %s %s/_out.gif'%(pltsearch, pltdir)
        os.system(cmd)
        X = data['acn_uq'][inds]
        X = X.reshape(X.shape[0], -1)
        #km = KMeans(n_clusters=max([1,inds.shape[0]//3]))
        #y = km.fit_predict(latents['st'][inds,0,0])
        # color points based on clustering, label, or index
        color = inds
        perplexity = 10
        param_name = '_tsne_%s_P%s.html'%(phase, perplexity)
        html_path = load_path.replace('.pt', param_name)
        tsne_plot(X=X, images=images, color=color,
                      perplexity=perplexity,
                      html_out_path=html_path, serve=False, img_size=cluster_img_size)
        param_name = '_pca_%s.html'%(phase)
        html_path = load_path.replace('.pt', param_name)
        pca_plot(X=X, images=images, color=color,
                     html_out_path=html_path, serve=False, img_size=cluster_img_size) 
        

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
    parser.add_argument('--rec_weight', type=float, default=4000)
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
        exp_desc = 'lin' 
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


    valid_fname = 'datasets/test_TD3_jaco_00000_0001140000_eval_CAM-1_S_S1140000_E0010_position.npz'
    train_fname = 'datasets/test_TD3_jaco_00000_0001160000_eval_CAM-1_S_S1160000_E0100_position.npz'
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

        latent_base = load_path.replace('.pt', '_latent_')
        model_dict = torch.load(load_path)
        losses = torch.load(load_path.replace('.pt', '.losses'))
        model.load_state_dict(model_dict['model'])
        prior_model.load_state_dict(model_dict['prior_model'])
        train_cnt = model_dict['train_cnt']
    if args.plot:
        plot_losses()
        for phase in ['valid', 'train']:
            create_latent_file(phase, latent_base+phase)
        print('plot latents')
        plot_latents(latent_base+'train.npz', latent_base+'valid.npz')
    else:
        mse_loss = nn.MSELoss(reduction='mean')
        parameters = []
        parameters+=list(model.parameters())
        parameters+=list(prior_model.parameters())
        opt = optim.Adam(parameters, lr=1e-6)
        run(train_cnt)

