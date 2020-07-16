from IPython import embed
import pickle
import numpy as np
from collections import OrderedDict
import torch
from torch import nn, optim
from torchvision import transforms
from torch.nn.utils.clip_grad import clip_grad_value_
from functions import vq, vq_st
from acn_models import tPTPriorNetwork
from acn_utils import kl_loss_function

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


def run():
    log_ones = torch.zeros(batch_size, code_length).to(device)
    train_cnt = 0
    losses = {
              'train':{'kl':[],'rec':[],'vq':[],'commit':[],'steps':[]},
              'valid':{'kl':[],'rec':[],'vq':[],'commit':[],'steps':[]}
              }
    while True:
        for phase in ['valid', 'train']:
            for i in range(replay_buffer[phase].size//batch_size):
                sample,batch_indexes = replay_buffer[phase].sample(batch_size, return_indexes=True)    
                st,ac,r,nst,nd,fr,nfr = sample
                rel_st = torch.FloatTensor(nst[:,3:3+7]-st[:,3:3+7])[:,None,None,:].to(device)
                state = torch.cat((rel_st,rel_st,rel_st,rel_st),3)
                u_q = model(state)
                u_q_flat = u_q.view(batch_size, code_length).contiguous()
                #if phase == 'train':
                #    assert batch_indexes.max() < prior_model.codes.shape[0]
                #    prior_model.update_codebook(batch_indexes, u_q_flat.detach())
                u_p, s_p = prior_model(u_q_flat)
                rec_state, z_e_x, z_q_x, latents = model.decode(u_q)
                kl = kl_loss_function(u_q.view(batch_size, code_length), 
                                      log_ones,
                                      u_p.view(batch_size, code_length), 
                                      s_p.view(batch_size, code_length),
                                      reduction='sum')
                rec_loss = mse_loss(rel_st, rec_state) 
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
                losses[phase]['rec'].append(kl.detach().cpu().item())
                losses[phase]['vq'].append(kl.detach().cpu().item())
                losses[phase]['commit'].append(kl.detach().cpu().item())
                losses[phase]['steps'].append(train_cnt)
            print(losses)

            model_dict = {'model':model.state_dict(), 
                          'prior_model':prior_model.state_dict(), 
                          'train_cnt':train_cnt, 
                          'losses':losses
                        }
            mpath = 'models/model_%08d.pt'%train_cnt
            torch.save(model_dict, mpath)
            print('saving {}'.format(mpath))
    embed() 

if __name__ == '__main__':
    valid_fname = 'results/relpos00/relpos_TD3_jaco_00000_0003100000_eval/test_TD3_jaco_00000_0003100000_eval_S3100000.epkl' 
    train_fname = 'results/relpos00/relpos_TD3_jaco_00000_0000040000.pkl'
    replay_buffer = {}
    replay_buffer['train'] = pickle.load(open(train_fname, 'rb'))
    replay_buffer['valid'] = pickle.load(open(valid_fname, 'rb'))
    device = 'cuda:0'
    batch_size = 128
    code_length = 112
    vq_commitment_beta = 0.25
    num_k = 5
    mse_loss = nn.MSELoss(reduction='sum')
    model = ACNresAngle().to(device)
    prior_model = tPTPriorNetwork(size_training_set=replay_buffer['train'].size, 
                                  code_length=code_length,
                                  k=num_k).to(device)
    parameters = []
    parameters+=list(model.parameters())
    parameters+=list(prior_model.parameters())
    opt = optim.Adam(parameters, lr=1e-4)
    run()

