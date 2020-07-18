import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys
from IPython import embed
import pickle
import numpy as np
from collections import OrderedDict
from copy import deepcopy
import torch
from torch import nn, optim
from torchvision import transforms
from torch.nn.utils.clip_grad import clip_grad_value_
from functions import vq, vq_st
from acn_models import tPTPriorNetwork
from acn_utils import kl_loss_function
from acn_utils import tsne_plot
from acn_utils import pca_plot
from sklearn.cluster import KMeans

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)

class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_).contiguous()
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar


# rithesh version
# https://github.com/ritheshkumar95/pytorch-vqvae/blob/master/vqvae.py
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class ACNresAngle(nn.Module):
    def __init__(self, input_size=1, num_clusters=256):
        super(ACNresAngle, self).__init__()
        num_z = 16
        bc = 1
        hs = 64
        # bs,1,1,states
        self.conv0 = nn.Conv2d(input_size, 16,  kernel_size=(1,1), stride=(1,1), padding=(0,0)) 
        self.conv1 = nn.Conv2d(16, 32, kernel_size=(1,4), stride=(1,2), padding=(0,1)) # --> 32,1,14
        self.conv2 = nn.Conv2d(32, hs, kernel_size=(1,4), stride=(1,2), padding=(0,1)) # --> 64,1,7
        self.conv3 = nn.Conv2d(hs, 32, kernel_size=(1,1), stride=(1,1), padding=(0,0)) # --> 16,1,7
        self.conv4 = nn.Conv2d(32, 16, kernel_size=(1,1), stride=(1,1), padding=(0,0)) # --> 16,1,7

  
        self.encoder = nn.Sequential(
                         OrderedDict([
                            ('0conv', self.conv0), 
                            ('1conv', self.conv1), 
                            ('1bn', nn.BatchNorm2d(32)), 
                            ('1relu', nn.ReLU(True)), 
                            ('2conv', self.conv2), 
                            ('3resblock', ResBlock(hs)), 
                            ('4conv', self.conv3), 
                            ('5conv', self.conv4)
                            ]))

        self.conv5 = nn.Conv2d(16, 32, kernel_size=(1,1), stride=(1,1), padding=(0,0)) # --> 32,1,7
        self.conv6 = nn.Conv2d(32, hs, kernel_size=(1,1), stride=(1,1), padding=(0,0)) # --> 64,1,7
        self.conv_layers = nn.Sequential(
                                         self.conv5,
                                         self.conv6,
                                         ResBlock(hs), 
                                         ResBlock(hs)
                                         )

        self.codebook = VQEmbedding(num_clusters, hs)
        # bs,1,1,states
        self.decoder = nn.Sequential(
                                     ResBlock(hs), 
                                     nn.BatchNorm2d(hs), 
                                     nn.ReLU(True), 
                                     nn.ConvTranspose2d(hs, 32,  kernel_size=(1,1), stride=(1,1), padding=(0,0)), #--> 32
                                     nn.BatchNorm2d(32), 
                                     nn.ReLU(True), 
                                     nn.ConvTranspose2d(32, 16,  kernel_size=(1,1), stride=(1,1), padding=(0,0)), #--> 32
                                     nn.ConvTranspose2d(16, input_size,  kernel_size=(1,1), stride=(1,1), padding=(0,0)), #--> 32
                                    )
    
    def forward(self, states):
        mu = self.encoder(states)
        if self.training:
            return mu+torch.randn(mu.shape).to(mu.device)
        else: 
            return mu
       
    def vq_encode(self, mu):
        z_e_x = self.conv_layers(mu) # 64,1,7
        latents = self.codebook(z_e_x)
        return z_e_x, latents

    def decode(self, mu):
        z_e_x, latents = self.vq_encode(mu)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st).contiguous()
        return x_tilde, z_e_x, z_q_x, latents


def run(train_cnt):
    log_ones = torch.zeros(batch_size, code_length).to(device)
    for dataset_view in range(10000):
        for phase in ['valid', 'train']:
            for i in range(replay_buffer[phase].size//batch_size):
                sample,batch_indexes = replay_buffer[phase].sample(batch_size, return_indexes=True)    
                st,ac,r,nst,nd,fr,nfr = sample
                rel_st = torch.FloatTensor(nst[:,3:3+7]-st[:,3:3+7])[:,None,None,:].to(device)
                state = torch.cat((rel_st,rel_st,rel_st,rel_st),3)
                u_q = model(state)
                u_q_flat = u_q.view(batch_size, code_length).contiguous()
                if phase == 'train':
                    assert batch_indexes.max() < prior_model.codes.shape[0]
                    prior_model.update_codebook(batch_indexes, u_q_flat.detach())
                u_p, s_p = prior_model(u_q_flat)
                rec_state, z_e_x, z_q_x, latents = model.decode(u_q)
                kl = kl_loss_function(u_q.view(batch_size, code_length), 
                                      log_ones,
                                      u_p.view(batch_size, code_length), 
                                      s_p.view(batch_size, code_length),
                                      reduction='sum')
                rec_loss = mse_loss(rel_st, rec_state)*(rec_state.shape[-1]*10000)
                vq_loss = mse_loss(z_q_x, z_e_x.detach())
                commit_loss = mse_loss(z_e_x, z_q_x.detach())*vq_commitment_beta
                if phase == 'train':
                    clip_grad_value_(model.parameters(), 10)
                    clip_grad_value_(prior_model.parameters(), 10)
                    kl.backward(retain_graph=True)
                    rec_loss.backward(retain_graph=True) 
                    vq_loss.backward(retain_graph=True)
                    commit_loss.backward(retain_graph=True)
                    opt.step()
                    train_cnt += batch_size 
            losses[phase]['kl'].append(kl.detach().cpu().item())
            losses[phase]['rec'].append(rec_loss.detach().cpu().item())
            losses[phase]['vq'].append(vq_loss.detach().cpu().item())
            losses[phase]['commit'].append(commit_loss.detach().cpu().item())
            losses[phase]['steps'].append(train_cnt)

            if phase == 'train' and not dataset_view%30:
                model_dict = {'model':model.state_dict(), 
                              'prior_model':prior_model.state_dict(), 
                              'train_cnt':train_cnt, 
                              'losses':losses
                            }
                mpath = 'models1/model_%012d.pt'%train_cnt
                torch.save(model_dict, mpath)
                print('saving {}'.format(mpath))
    embed() 

def create_latent_file():
    model.eval()
    prior_model.eval()
    latent_base = load_path.replace('.pt', '_latents')
    blues = plt.cm.Blues(np.linspace(0.2,.8,num_k))[::-1]
    pltdir = load_path.replace('.pt', '_plots')
    for phase in ['train', 'valid']:
        if not os.path.exists(latent_base + '_%s.npz'%phase):
            sz = replay_buffer[phase].size
            all_indexes = []; all_st = []
            all_rel_st = []; all_rec_st = []
            all_acn_uq = []; all_acn_sp = []
            all_neighbors = []; all_neighbor_distances = []; all_vq_latents = []
            for i in range(0,sz,batch_size):
                batch_indexes = np.arange(i, min([i+batch_size, sz]))
                sample = replay_buffer[phase].get_indexes(batch_indexes)    
                st,ac,r,nst,nd,fr,nfr = sample
                all_st.extend(st[:,3:3+7])
                rel_st = torch.FloatTensor(nst[:,3:3+7]-st[:,3:3+7])[:,None,None,:].to(device)
                state = torch.cat((rel_st,rel_st,rel_st,rel_st),3)
                u_q = model(state)
                u_q_flat = u_q.view(batch_indexes.shape[0], code_length).contiguous()
                u_p, s_p = prior_model(u_q_flat)
                rec_st, z_e_x, z_q_x, latents = model.decode(u_q)
                neighbor_distances, neighbor_indexes = prior_model.kneighbors(u_q_flat, n_neighbors=num_k)
                all_indexes.extend(batch_indexes)
                all_rel_st.extend(deepcopy(rel_st.detach().cpu().numpy()))
                all_rec_st.extend(deepcopy(rec_st.detach().cpu().numpy()))
                all_acn_uq.extend(deepcopy(u_q.detach().cpu().numpy()))
                all_acn_sp.extend(deepcopy(s_p.detach().cpu().numpy()))
                all_neighbors.extend(deepcopy(neighbor_indexes.detach().cpu().numpy()))
                all_neighbor_distances.extend(deepcopy(neighbor_distances.detach().cpu().numpy()))
                all_vq_latents.extend(latents.detach().cpu().numpy())
                if i == 0 and phase=='train':
                    print(all_neighbors)
            np.savez(latent_base+'_%s'%phase,
                     index=all_indexes,
                     st=all_st,
                     rel_st=all_rel_st,
                     rec_st=all_rec_st,
                     acn_uq=all_acn_uq,
                     acn_sp=all_acn_sp,
                     neighbor_train_indexes=all_neighbors,
                     neighbor_distances=all_neighbor_distances,
                     vq_latents=all_vq_latents)
    return latent_base


def plot_losses():
    plt_base = load_path.replace('.pt', '')
    for lparam in losses['train'].keys():
        if lparam != 'steps':
            plt.figure()
            plt.plot(losses['train']['steps'][1:], losses['train'][lparam][1:], label='train')
            plt.plot(losses['valid']['steps'][1:], losses['valid'][lparam][1:], label='valid')
            plt.title(lparam)
            plt.legend()
            plt.savefig(plt_base+'_losses_%s.png'%lparam)
            plt.close()

def plot_latents(train_latent_file, valid_latent_file):
    
    train_data = np.load(train_latent_file)
    valid_data = np.load(valid_latent_file)
    blues = plt.cm.Blues(np.linspace(0.2,.8,num_k))[::-1]

    pltdir = load_path.replace('.pt', '_plots')
    if not os.path.exists(pltdir):
        os.makedirs(pltdir)
    
    ymin = -.1 #train_latents['rel_st'].min()
    ymax = .1  #train_latents['rel_st'].max()
    for phase, data in [('train',train_data), ('valid',valid_data)]:
        images = []
        #inds = np.random.randint(0, data['rel_st'].shape[0], args.num_plot_examples).astype(np.int)
        inds = np.arange(0, min([400,data['rel_st'].shape[0]])).astype(np.int)
        for i in inds:
            plt.figure()
            plt.plot(data['rel_st'][i,0,0], label='data', lw=2, marker='x', color='mediumorchid')
            plt.plot(data['rec_st'][i,0,0], label='rec', lw=1.5, marker='o', color='blue')
            #plt.plot(data['rel_st'][i,0,0] + data['st'][i], label='data', lw=2, marker='x', color='mediumorchid')
            #plt.plot(data['rec_st'][i,0,0] + data['st'][i], label='rec', lw=1.5, marker='o', color='blue')
            for nn, n in enumerate(data['neighbor_train_indexes'][i]):
                # relative to train latents
                #plt.plot(train_data['rel_st'][n,0,0] + data['st'][i], label='k:%d i:%d'%(nn,n), lw=1., marker='.', color=blues[nn], alpha=.5)
                plt.plot(train_data['rel_st'][n,0,0], label='k:%d i:%d'%(nn,n), lw=1., marker='.', color=blues[nn], alpha=.5)
            plt.ylim(ymin, ymax)                    
            plt.title('%s'%i)
            #plt.legend()
            pltname = os.path.join(pltdir, '%s_%05d.png'%(phase,i))
            print('plotting %s'%pltname)
            plt.savefig(pltname)
            plt.close()
     
            images.append(plt.imread(pltname))
        images = np.array(images)
        X = data['acn_uq'][inds]
        X = X.reshape(X.shape[0], -1)
        #km = KMeans(n_clusters=max([1,inds.shape[0]//3]))
        #y = km.fit_predict(latents['rel_st'][inds,0,0])
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', default=False, action='store_true')
    parser.add_argument('--load_model', default='')
    parser.add_argument('--num_plot_examples', default=500)
    cluster_img_size = (120,120)
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit()

    valid_fname = 'results/relpos00/relpos_TD3_jaco_00000_0003100000_eval/test_TD3_jaco_00000_0003100000_eval_S3100000.epkl' 
    train_fname = 'results/relpos00/relpos_TD3_jaco_00000_0000060000.pkl'
    replay_buffer = {}
    replay_buffer['train'] = pickle.load(open(train_fname, 'rb'))
    replay_buffer['valid'] = pickle.load(open(valid_fname, 'rb'))
    device = 'cuda:0'
    batch_size = 128
    code_length = 112
    vq_commitment_beta = 0.25
    num_k = 5
    losses = {
              'train':{'kl':[],'rec':[],'vq':[],'commit':[],'steps':[]},
              'valid':{'kl':[],'rec':[],'vq':[],'commit':[],'steps':[]}
              }
 
    model = ACNresAngle().to(device)
    train_size = replay_buffer['train'].size
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
        model_dict = torch.load(load_path)
        losses = model_dict['losses']
        model.load_state_dict(model_dict['model'])
        prior_model.load_state_dict(model_dict['prior_model'])
        train_cnt = model_dict['train_cnt']
    if args.plot:
        plot_losses()
        latent_base = create_latent_file()
        plot_latents(latent_base + '_train.npz', latent_base + '_valid.npz')
    else:
        mse_loss = nn.MSELoss(reduction='sum')
        parameters = []
        parameters+=list(model.parameters())
        parameters+=list(prior_model.parameters())
        opt = optim.Adam(parameters, lr=1e-4)
        run(train_cnt)

