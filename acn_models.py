import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from torch.nn import functional as F
import torch
# fast vq from rithesh
from functions import vq, vq_st
from IPython import embed
from collections import OrderedDict

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
    else:
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
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar

class VQLinearEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        latents = vq(z_e_x.contiguous(), self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x = z_e_x.contiguous()
        z_q_x_, indices = vq_st(z_e_x, self.embedding.weight.detach())
        z_q_x = z_q_x_.contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight, dim=0, index=indices)
        z_q_x_bar = z_q_x_bar_flatten.view_as(z_e_x).contiguous()
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
                            ('3conv', self.conv3), 
                            ('4conv', self.conv4)
                            ]))

        #self.conv5 = nn.Conv2d(16, 32, kernel_size=(1,1), stride=(1,1), padding=(0,0)) # --> 32,1,7
        #self.conv6 = nn.Conv2d(32, hs, kernel_size=(1,1), stride=(1,1), padding=(0,0)) # --> 64,1,7
        #self.conv_layers = nn.Sequential(
        #                                 self.conv5,
        #                                 self.conv6,
        #                                 ResBlock(hs), 
        #                                 ResBlock(hs)
        #                                 )

        #self.codebook = VQEmbedding(num_clusters, hs)
        # bs,1,1,states
        self.acn_decoder = nn.Sequential(
                                     ResBlock(16), 
                                     nn.BatchNorm2d(16), 
                                     nn.ReLU(True), 
                                     nn.ConvTranspose2d(16, 16,  kernel_size=(1,1), stride=(1,1), padding=(0,0)), #--> 32
                                     nn.BatchNorm2d(16), 
                                     nn.ReLU(True), 
                                     nn.ConvTranspose2d(16, 16,  kernel_size=(1,1), stride=(1,1), padding=(0,0)), #--> 32
                                     nn.ConvTranspose2d(16, input_size,  kernel_size=(1,1), stride=(1,1), padding=(0,0)), #--> 32
                                    )
    
    def forward(self, states):
        mu = self.encoder(states)
        if self.training:
            return mu+torch.randn(mu.shape).to(mu.device)
        else: 
            return mu
       
    def acn_decode(self, mu):
        x_tilde = self.acn_decoder(mu).contiguous()
        return x_tilde
#    def vq_encode(self, mu):
#        z_e_x = self.conv_layers(mu) # 64,1,7
#        latents = self.codebook(z_e_x)
#        return z_e_x, latents
#
#    def decode(self, mu):
#        z_e_x, latents = self.vq_encode(mu)
#        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
#        x_tilde = self.decoder(z_q_x_st).contiguous()
#        return x_tilde, z_e_x, z_q_x, latents
#


class ACNstateAngle(nn.Module):
    def __init__(self, input_size=7, num_clusters=256, code_length=32, hidden_size=512):
        super(ACNstateAngle, self).__init__()
        hs = hidden_size
        self.encoder = nn.Sequential(
                         OrderedDict([
                            ('lin0',nn.Linear(input_size, hs)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin1',nn.Linear(hs, hs)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin2',nn.Linear(hs, code_length))]
                           ))
        self.decoder = nn.Sequential(
                         OrderedDict([
                            ('lin0',nn.Linear(code_length, hs)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin1',nn.Linear(hs, hs)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin2',nn.Linear(hs, input_size))]
                           ))
 
    def forward(self, states):
        mu = self.encoder(states)
        if self.training:
            noise = (torch.randn(mu.shape)).to(mu.device)
            return mu+noise
        else: 
            return mu

    def acn_decode(self, mu):
        x_tilde = self.decoder(mu).contiguous()
        return x_tilde


class ACNstateAngleBig(nn.Module):
    def __init__(self, input_size=7, num_clusters=256, code_length=32, hidden_size=512):
        super(ACNstateAngleBig, self).__init__()
        hs = hidden_size
        self.encoder = nn.Sequential(
                         OrderedDict([
                            ('lin0',nn.Linear(input_size, 256)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin1',nn.Linear(256, hs)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin2',nn.Linear(hs, hs)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin3',nn.Linear(hs, 256)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin4',nn.Linear(256, code_length))]
                           ))
        self.decoder = nn.Sequential(
                         OrderedDict([
                            ('lin0',nn.Linear(code_length, 256)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin1',nn.Linear(256, hs)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin2',nn.Linear(hs, hs)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin3',nn.Linear(hs, 256)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin4',nn.Linear(256, input_size))]
                           ))
 
    def forward(self, states):
        mu = self.encoder(states)
        if self.training:
            noise = (torch.randn(mu.shape)).to(mu.device)
            return mu+noise
        else: 
            return mu

    def acn_decode(self, mu):
        x_tilde = self.decoder(mu).contiguous()
        return x_tilde





class ACNstateAngleEmbedding(nn.Module):
    def __init__(self, n_bins, input_size=7, num_clusters=256, code_length=32, hidden_size=1024, embedding_dim=128):
        super(ACNstateAngleEmbedding, self).__init__()
        hs = hidden_size
        self.embeddings = nn.Embedding(n_bins, embedding_dim)

        self.encoder = nn.Sequential(
                         OrderedDict([
                            ('lin0',nn.Linear(input_size*embedding_dim, hs)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin1',nn.Linear(hs, hs)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin2',nn.Linear(hs, hs)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin3',nn.Linear(hs, 256)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin4',nn.Linear(256, code_length))]
                           ))
        self.decoder = nn.Sequential(
                         OrderedDict([
                            ('lin0',nn.Linear(code_length, 256)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin1',nn.Linear(256, hs)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin2',nn.Linear(hs, hs)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin3',nn.Linear(hs, 256)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin4',nn.Linear(256, input_size))]
                           ))
 
    def forward(self, states):
        embeds = self.embeddings(states).view((states.shape[0], -1))  
        mu = self.encoder(embeds)
        if self.training:
            noise = (torch.randn(mu.shape)).to(mu.device)
            return mu+noise
        else: 
            return mu

    def acn_decode(self, mu):
        x_tilde = self.decoder(mu).contiguous()
        return x_tilde





class ACNVQVAEresMNISTsmall(nn.Module):
    def __init__(self, code_len, input_size=1, output_size=1,
                       hidden_size=256,
                 num_clusters=512, num_z=32):

        super(ACNVQVAEresMNISTsmall, self).__init__()
        self.code_len = code_len
        self.num_clusters = num_clusters
        self.num_z = num_z
        self.hidden_size = hidden_size
        # encoder output size found experimentally when architecture changes
        self.eo = 7
        fsos = first_stage_output_size = 128
        encoder_match_size = ems = hidden_size
        bc = self.bottleneck_channels = 2
        self.encoder = nn.Sequential(
                nn.Conv2d(input_size, hidden_size, 4, 2, 1),
                nn.BatchNorm2d(hidden_size),
                nn.ReLU(True),
                nn.Conv2d(hidden_size, hidden_size, 4, 2, 1),
                ResBlock(hidden_size),
                ResBlock(hidden_size),
                nn.Conv2d(hidden_size, 16, 1, 1, 0),
                nn.Conv2d(16, bc, 1, 1, 0),
                )
        self.decoder = nn.Sequential(
                  ResBlock(hidden_size),
                  ResBlock(hidden_size),
                  nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 1),
                  nn.BatchNorm2d(hidden_size),
                  nn.ReLU(True),
                  nn.ConvTranspose2d(hidden_size, output_size, 4, 2, 1),
                  )

        self.conv_layers = nn.Sequential(
            nn.Conv2d(bc, 16, 1, 1, 0),
            nn.Conv2d(16, hidden_size, 1, 1, 0),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.Conv2d(hidden_size, hidden_size, 1, 1, 0),
            ResBlock(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 1, 1, 0),
            )
        self.codebook = VQEmbedding(num_clusters, hidden_size)
        self.apply(weights_init)

    def reparameterize(self, mu):
        if self.training:
            noise = torch.randn(mu.shape).to(mu.device)
            return mu+noise
        else:
            return mu

    def vq_encode(self, mu):
        z_e_x = self.conv_layers(mu)
        latents = self.codebook(z_e_x)
        return z_e_x, latents

    def forward(self, frames):
        # x is bs,hidden_size,h,w
        # mu is 256,1,11,11
        mu = self.encoder(frames)
        z = self.reparameterize(mu)
        return z, mu

    def decode(self, z):
        z_e_x, latents = self.vq_encode(z)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x, latents



class ACNVQVAEresMNIST(nn.Module):
    def __init__(self, code_len, input_size=1, output_size=1,
                       hidden_size=256,
                 num_clusters=512, num_z=32):

        super(ACNVQVAEresMNIST, self).__init__()
        self.code_len = code_len
        self.num_clusters = num_clusters
        self.num_z = num_z
        self.hidden_size = hidden_size
        # encoder output size found experimentally when architecture changes
        self.eo = 7
        fsos = first_stage_output_size = 128
        encoder_match_size = ems = hidden_size
        bc = self.bottleneck_channels = 2
        self.encoder = nn.Sequential(
                nn.Conv2d(input_size, hidden_size, 4, 2, 1),
                nn.BatchNorm2d(hidden_size),
                nn.ReLU(True),
                nn.Conv2d(hidden_size, hidden_size, 4, 2, 1),
                ResBlock(hidden_size),
                ResBlock(hidden_size),
                nn.Conv2d(hidden_size, 16, 1, 1, 0),
                nn.Conv2d(16, bc, 1, 1, 0),
                )
        self.decoder = nn.Sequential(
                  ResBlock(hidden_size),
                  ResBlock(hidden_size),
                  nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 1),
                  nn.BatchNorm2d(hidden_size),
                  nn.ReLU(True),
                  nn.ConvTranspose2d(hidden_size, output_size, 4, 2, 1),
                  )

        self.conv_layers = nn.Sequential(
            nn.Conv2d(bc, 16, 1, 1, 0),
            nn.Conv2d(16, hidden_size, 1, 1, 0),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.Conv2d(hidden_size, hidden_size, 1, 1, 0),
            ResBlock(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 1, 1, 0),
            )
        self.codebook = VQEmbedding(num_clusters, hidden_size)
        self.apply(weights_init)

    def reparameterize(self, mu):
        if self.training:
            noise = torch.randn(mu.shape).to(mu.device)
            return mu+noise
        else:
            return mu

    def vq_encode(self, mu):
        z_e_x = self.conv_layers(mu)
        latents = self.codebook(z_e_x)
        return z_e_x, latents

    def forward(self, frames):
        # x is bs,hidden_size,h,w
        # mu is 256,1,11,11
        mu = self.encoder(frames)
        z = self.reparameterize(mu)
        return z, mu

    def decode(self, z):
        z_e_x, latents = self.vq_encode(z)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x, latents

class ACNresMNIST(nn.Module):
    def __init__(self, code_len, input_size=1, output_size=1,
                       hidden_size=256,
                 ):

        super(ACNresMNIST, self).__init__()
        self.code_len = code_len
        self.hidden_size = hidden_size
        # encoder output size found experimentally when architecture changes
        self.eo = 7
        fsos = first_stage_output_size = 128
        encoder_match_size = ems = hidden_size
        self.frame_encoder = nn.Sequential(
                               nn.Conv2d(input_size, fsos, 1, 1, 0),
                               nn.BatchNorm2d(fsos),
                               nn.ReLU(True),
                               nn.Conv2d(fsos, fsos, 1, 1, 0),
                               nn.ReLU(True),
                             )
        bc = self.bottleneck_channels = 2
        self.encoder = nn.Sequential(
                nn.Conv2d(fsos, hidden_size, 4, 2, 1),
                nn.BatchNorm2d(hidden_size),
                nn.ReLU(True),
                nn.Conv2d(hidden_size, hidden_size, 4, 2, 1),
                ResBlock(hidden_size),
                ResBlock(hidden_size),
                nn.Conv2d(hidden_size, 16, 1, 1, 0),
                nn.Conv2d(16, bc, 1, 1, 0),
                )
        self.decoder = nn.Sequential(
                  nn.ConvTranspose2d(bc, 16, 1, 1, 0),
                  nn.ConvTranspose2d(16, hidden_size, 1, 1, 0),
                  ResBlock(hidden_size),
                  ResBlock(hidden_size),
                  nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 1),
                  nn.BatchNorm2d(hidden_size),
                  nn.ReLU(True),
                  nn.ConvTranspose2d(hidden_size, output_size, 4, 2, 1),
                  )

        self.apply(weights_init)

    def reparameterize(self, mu):
        if self.training:
            noise = torch.randn(mu.shape).to(mu.device)
            return mu+noise
        else:
            return mu

    def forward(self, frames):
        x = self.frame_encoder(frames)
        mu = self.encoder(x)
        z = self.reparameterize(mu)
        return z, mu

    def decode(self, z):
        x_tilde = self.decoder(z)
        return x_tilde



class ACNVQVAEres(nn.Module):
    def __init__(self, code_len, input_size=1, output_size=1,
                       hidden_size=256,
                 num_clusters=512, num_z=32):

        super(ACNVQVAEres, self).__init__()
        self.code_len = code_len
        self.num_clusters = num_clusters
        self.num_z = num_z
        self.hidden_size = hidden_size
        # encoder output size found experimentally when architecture changes
        self.eo = 8
        fsos = first_stage_output_size = 128
        encoder_match_size = ems = hidden_size
        self.frame_encoder = nn.Sequential(
                               nn.Conv2d(input_size, fsos, 1, 1, 0),
                               nn.BatchNorm2d(fsos),
                               nn.ReLU(True),
                               nn.Conv2d(fsos, fsos, 1, 1, 0),
                               nn.ReLU(True),
                             )
        bc = self.bottleneck_channels = 1
        self.encoder = nn.Sequential(
                nn.Conv2d(fsos, hidden_size, 4, 2, 1),
                nn.BatchNorm2d(hidden_size),
                nn.ReLU(True),
                nn.Conv2d(hidden_size, hidden_size, 4, 2, 0),
                nn.Conv2d(hidden_size, hidden_size, 4, 2, 0),
                nn.Conv2d(hidden_size, hidden_size, 2, 1, 0),
                ResBlock(hidden_size),
                ResBlock(hidden_size),
                nn.Conv2d(hidden_size, 16, 1, 1, 0),
                nn.Conv2d(16, bc, 1, 1, 0),
                )
        self.decoder = nn.Sequential(
                  ResBlock(hidden_size),
                  ResBlock(hidden_size),
                  nn.ConvTranspose2d(hidden_size, hidden_size, 2, 1, 0),
                  nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 0),
                  nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 0),
                  nn.BatchNorm2d(hidden_size),
                  nn.ReLU(True),
                  nn.ConvTranspose2d(hidden_size, output_size, 4, 2, 1),
                  )

        self.conv_layers = nn.Sequential(
            nn.Conv2d(bc, 16, 1, 1, 0),
            nn.Conv2d(16, hidden_size, 1, 1, 0),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
            nn.Conv2d(hidden_size, hidden_size, 1, 1, 0),
            ResBlock(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 1, 1, 0),
            )
        self.codebook = VQEmbedding(num_clusters, hidden_size)
        self.apply(weights_init)

    def reparameterize(self, mu):
        if self.training:
            noise = torch.randn(mu.shape).to(mu.device)
            return mu+noise
        else:
            return mu

    def vq_encode(self, mu):
        z_e_x = self.conv_layers(mu)
        latents = self.codebook(z_e_x)
        return z_e_x, latents

    def forward(self, frames):
        x = self.frame_encoder(frames)
        mu = self.encoder(x)
        z = self.reparameterize(mu)
        return z, mu

    def decode(self, z):
        z_e_x, latents = self.vq_encode(z)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x, latents

class tPTPriorNetwork(nn.Module):
    def __init__(self, size_training_set, code_length, n_hidden=512, k=5, random_seed=4543):
        super(tPTPriorNetwork, self).__init__()
        self.rdn = np.random.RandomState(random_seed)
        self.k = k
        self.size_training_set = size_training_set
        self.code_length = code_length
        self.input_layer = nn.Linear(code_length, n_hidden)
        self.skipin_to_2 = nn.Linear(n_hidden, n_hidden)
        self.skipin_to_3 = nn.Linear(n_hidden, n_hidden)
        self.skip1_to_out = nn.Linear(n_hidden, n_hidden)
        self.skip2_to_out = nn.Linear(n_hidden, n_hidden)
        self.h1 = nn.Linear(n_hidden, n_hidden)
        self.h2 = nn.Linear(n_hidden, n_hidden)
        self.h3 = nn.Linear(n_hidden, n_hidden)
        self.fc_mu = nn.Linear(n_hidden, self.code_length)
        self.fc_s = nn.Linear(n_hidden, self.code_length)

        # needs to be a param so that we can load
        self.codes = nn.Parameter(torch.FloatTensor(self.rdn.standard_normal((self.size_training_set, self.code_length))), requires_grad=False)
        # start off w/ default batch size - this will change automatically if
        # different input is given
        batch_size = 64
        n_neighbors = 5
        self.neighbors = torch.LongTensor((batch_size, n_neighbors))
        self.distances = torch.FloatTensor((batch_size, n_neighbors))
        self.batch_indexer = torch.LongTensor(torch.arange(batch_size))

    def update_codebook(self, indexes, values):
        assert indexes.min() >= 0
        assert indexes.max() < self.codes.shape[0]
        self.codes[indexes] = values

    def kneighbors(self, test, n_neighbors):
        with torch.no_grad():
            device = test.device
            bs = test.shape[0]
            return_size = (bs,n_neighbors)
            # dont recreate unless necessary
            if self.neighbors.shape != return_size:
                print('updating prior sizes')
                self.neighbors = torch.LongTensor(torch.zeros(return_size, dtype=torch.int64))
                self.distances = torch.zeros(return_size)
                self.batch_indexer = torch.LongTensor(torch.arange(bs))
            if device != self.codes.device:
                print('transferring prior to %s'%device)
                self.neighbors = self.neighbors.to(device)
                self.distances = self.distances.to(device)
                self.codes = self.codes.to(device)
            for bidx in range(test.shape[0]):
                dists = torch.norm(self.codes-test[bidx], dim=1)
                self.distances[bidx], self.neighbors[bidx] = dists.topk(n_neighbors, largest=False)
                del dists
        #print('kn', bidx, torch.cuda.memory_allocated(device=None))
        return self.distances.detach(), self.neighbors.detach()

    def batch_pick_close_neighbor(self, codes):
        '''
        :code latent activation of training
        '''
        neighbor_distances, neighbor_indexes = self.kneighbors(codes, n_neighbors=self.k)
        bsize = neighbor_indexes.shape[0]
        if self.training:
            # randomly choose neighbor index from top k
            chosen_neighbor_index = torch.LongTensor(self.rdn.randint(0,neighbor_indexes.shape[1],size=bsize))
        else:
            chosen_neighbor_index = torch.LongTensor(torch.zeros(bsize, dtype=torch.int64))
        return self.codes[neighbor_indexes[self.batch_indexer, chosen_neighbor_index]]

    def forward(self, codes):
        previous_codes = self.batch_pick_close_neighbor(codes)
        return self.encode(previous_codes)

    def encode(self, prev_code):
        """
        The prior network was an
        MLP with three hidden layers each containing 512 tanh
        units
        - and skip connections from the input to all hidden
        layers and
        - all hiddens to the output layer.
        """
        i = torch.tanh(self.input_layer(prev_code))
        # input goes through first hidden layer
        _h1 = torch.tanh(self.h1(i))

        # make a skip connection for h layers 2 and 3
        _s2 = torch.tanh(self.skipin_to_2(_h1))
        _s3 = torch.tanh(self.skipin_to_3(_h1))

        # h layer 2 takes in the output from the first hidden layer and the skip
        # connection
        _h2 = torch.tanh(self.h2(_h1+_s2))

        # take skip connection from h1 and h2 for output
        _o1 = torch.tanh(self.skip1_to_out(_h1))
        _o2 = torch.tanh(self.skip2_to_out(_h2))
        # h layer 3 takes skip connection from prev layer and skip connection
        # from nput
        _o3 = torch.tanh(self.h3(_h2+_s3))

        out = _o1+_o2+_o3
        mu = self.fc_mu(out)
        logstd = self.fc_s(out)
        return mu, logstd



