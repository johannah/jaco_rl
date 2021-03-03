from torch import nn
import torch
from IPython import embed
from collections import OrderedDict

class AngleAutoEncoder(nn.Module):
    def __init__(self, input_size=5, output_size=1, hidden_size=128):
        super(AngleAutoEncoder, self).__init__()
        hs = hidden_size
        # bs,1,1,states
        self.angle_network = nn.Sequential(
                         OrderedDict([
                            ('lin0',nn.Linear(input_size, hs)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin1',nn.Linear(hs, hs))]
                           ))
        self.target_position_network = nn.Sequential(
                         OrderedDict([
                            ('lin0',nn.Linear(3, hs)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin1',nn.Linear(hs, hs))]
                           ))
 
        self.output_angle_network = nn.Sequential(
                         OrderedDict([
                            ('lin0',nn.Linear(hs*2, hs)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin1',nn.Linear(hs, hs)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin2',nn.Linear(hs, output_size))]
                           ))
 
    def forward(self, angle, target):
        aout = self.angle_network(angle) 
        tout = self.target_position_network(target) 
        # angle outptu 
        return self.output_angle_network(torch.cat((tout,aout),1))



class BigAngleAutoEncoder(nn.Module):
    def __init__(self, input_size=5, output_size=1):
        super(BigAngleAutoEncoder, self).__init__()
        # bs,1,1,states
        self.angle_network = nn.Sequential(
                         OrderedDict([
                            ('lin0',nn.Linear(input_size, 32)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin1',nn.Linear(32, 64)),
                            ('relu',nn.ReLU(True)), 
                            ('lin2',nn.Linear(64, 128)),
                            ('relu',nn.ReLU(True)), 
                            ('lin3',nn.Linear(128, 512)),
                            ('relu',nn.ReLU(True)), 
                            ('lin4',nn.Linear(512, 512)),
                            ]))
                           
        self.target_position_network = nn.Sequential(
                         OrderedDict([
                            ('lin0',nn.Linear(3, 32)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin1',nn.Linear(32, 64)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin2',nn.Linear(64, 128)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin3',nn.Linear(128,512)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin4',nn.Linear(512, 512)), 
                            ]))
 
        self.output_angle_network = nn.Sequential(
                         OrderedDict([
                            ('lin0',nn.Linear(512*2, 512)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin1',nn.Linear(512, 512)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin2',nn.Linear(512, 128)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin3',nn.Linear(128, 64)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin4',nn.Linear(64, output_size))]
                           ))
 
    def forward(self, angle, target):
        aout = self.angle_network(angle) 
        tout = self.target_position_network(target) 
        # angle outptu 
        return self.output_angle_network(torch.cat((tout,aout),1))

class ResidualBlockLinear(nn.Module):
    def __init__(self, hidden_size=1024):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size), 
            nn.BatchNorm1d(hidden_size), 
            nn.ReLU(True), 
            nn.Linear(hidden_size, hidden_size), 
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(True), 
            )

    def forward(self, x): 
        return x + self.block(x)
            
           

class AngleResidual(nn.Module):
    def __init__(self, input_size=9, output_size=1):
        super(AngleResidual, self).__init__()
        # bs,1,1,states
        # 4 or 5 resnet layers and an output layer
        hidden_size = 1024
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.block1 = ResidualBlockLinear(hidden_size)
        self.block2 = ResidualBlockLinear(hidden_size)
        self.block3 = ResidualBlockLinear(hidden_size)
        self.block4 = ResidualBlockLinear(hidden_size)
        self.layerout = nn.Linear(hidden_size, output_size)

    def forward(self, x):
       x = self.block1(torch.relu(self.layer1(x))) 
       x = self.block2(x) 
       x = self.block3(x)
       x = self.layerout(self.block4(x))
       return x
   

class BigAngleSequential(nn.Module):
    def __init__(self, input_size=5, output_size=1):
        super(BigAngleAutoEncoder, self).__init__()
        # bs,1,1,states
        self.angle_network = nn.Sequential(
                         OrderedDict([
                            ('lin0',nn.Linear(input_size, 32)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin1',nn.Linear(32, 64)),
                            ('relu',nn.ReLU(True)), 
                            ('lin2',nn.Linear(64, 128)),
                            ('relu',nn.ReLU(True)), 
                            ('lin3',nn.Linear(128, 512)),
                            ('relu',nn.ReLU(True)), 
                            ('lin4',nn.Linear(512, 512)),
                            ]))
                           
        self.target_position_network = nn.Sequential(
                         OrderedDict([
                            ('lin0',nn.Linear(3, 32)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin1',nn.Linear(32, 64)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin2',nn.Linear(64, 128)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin3',nn.Linear(128,512)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin4',nn.Linear(512, 512)), 
                            ]))
 
        self.output_angle_network = nn.Sequential(
                         OrderedDict([
                            ('lin0',nn.Linear(512*2, 512)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin1',nn.Linear(512, 512)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin2',nn.Linear(512, 128)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin3',nn.Linear(128, 64)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin4',nn.Linear(64, output_size))]
                           ))
 
    def target_forward(target):
        return self.target_position_network(target) 

    def forward(self, angle, target_output):
        aout = self.angle_network(angle) 
        return self.output_angle_network(torch.cat((target_output, aout),1))


class AngleResidualEmbedBoth(nn.Module):
    def __init__(self, n_bins, n_target_bins, input_size=9, output_size=1, embedding_dim=128):
        super(AngleResidualEmbedBoth, self).__init__()
        # bs,1,1,states
        # 4 or 5 resnet layers and an output layer
        hidden_size = 1024
        self.embeddings = nn.Embedding(n_bins, embedding_dim)
        self.target_embeddings = nn.Embedding(n_target_bins, embedding_dim)
        size_embed_output = (input_size*embedding_dim) + (3*embedding_dim)
 
        self.layer1 = nn.Linear(size_embed_output, hidden_size)
        self.block1 = ResidualBlockLinear(hidden_size)
        self.block2 = ResidualBlockLinear(hidden_size)
        self.block3 = ResidualBlockLinear(hidden_size)
        self.block4 = ResidualBlockLinear(hidden_size)
        self.layerout = nn.Linear(hidden_size, output_size)

    def forward(self, angle, target):
        bs = angle.shape[0]
        embeds = self.embeddings(angle).view((bs, -1))
        tembeds = self.target_embeddings(target).view((bs, -1))
        x = torch.cat((embeds, tembeds), 1)
        x = self.block1(torch.relu(self.layer1(x))) 
        x = self.block2(x) 
        x = self.block3(x)
        x = self.layerout(self.block4(x))
        return x
   

class BigAngleEmbed(nn.Module):
    def __init__(self, n_bins, input_size=7, output_size=1, hidden_size=1024, embedding_dim=128):
        super(BigAngleEmbed, self).__init__()
        # bs,1,1,states
        hs = hidden_size
        self.embeddings = nn.Embedding(n_bins, embedding_dim)
        self.angle_network = nn.Sequential(
                         OrderedDict([
                            ('lin0',nn.Linear(input_size*embedding_dim, hs)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin1',nn.Linear(hs, hs)),
                            ('relu',nn.ReLU(True)), 
                            ('lin2',nn.Linear(hs, hs)),
                            ('relu',nn.ReLU(True)), 
                            ('lin3',nn.Linear(hs, hs)),
                            ('relu',nn.ReLU(True)), 
                            ('lin4',nn.Linear(hs, hs)),
                            ]))
                           
        self.target_position_network = nn.Sequential(
                         OrderedDict([
                            ('lin0',nn.Linear(3, 128)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin1',nn.Linear(128, 512)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin2',nn.Linear(512, hs)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin3',nn.Linear(hs, hs)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin4',nn.Linear(hs, hs)), 
                            ]))
 
        self.output_angle_network = nn.Sequential(
                         OrderedDict([
                            ('lin0',nn.Linear(hs*2, hs*2)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin1',nn.Linear(hs*2, hs)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin2',nn.Linear(hs, hs)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin3',nn.Linear(hs, hs)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin4',nn.Linear(hs, 512)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin5',nn.Linear(512, 256)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin6',nn.Linear(256, 64)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin7',nn.Linear(64, output_size))]
                           ))
 
    def forward(self, angle, target):
        aout = self.angle_network(embeds) 
        tout = self.target_position_network(target) 
        # angle outptu 
        return self.output_angle_network(torch.cat((tout,aout),1))


class BigAngleEmbedBoth(nn.Module):
    def __init__(self, n_bins, n_target_bins, input_size=7, output_size=1, hidden_size=1024, embedding_dim=128):
        super(BigAngleEmbedBoth, self).__init__()
        # bs,1,1,states
        hs = hidden_size
        self.embeddings = nn.Embedding(n_bins, embedding_dim)
        self.target_embeddings = nn.Embedding(n_target_bins, embedding_dim)
        self.angle_network = nn.Sequential(
                         OrderedDict([
                            ('lin0',nn.Linear(input_size*embedding_dim, hs)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin1',nn.Linear(hs, hs)),
                            ('relu',nn.ReLU(True)), 
                            ('lin2',nn.Linear(hs, hs)),
                            ('relu',nn.ReLU(True)), 
                            ('lin3',nn.Linear(hs, hs)),
                            ('relu',nn.ReLU(True)), 
                            ('lin4',nn.Linear(hs, hs)),
                            ]))
                           
        self.target_position_network = nn.Sequential(
                         OrderedDict([
                            ('lin0',nn.Linear(3*embedding_dim, hs)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin1',nn.Linear(hs, hs)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin3',nn.Linear(hs, hs)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin4',nn.Linear(hs, hs)), 
                            ]))
 
        self.output_angle_network = nn.Sequential(
                         OrderedDict([
                            ('lin0',nn.Linear(hs*2, hs*2)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin1',nn.Linear(hs*2, hs)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin4',nn.Linear(hs, 512)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin5',nn.Linear(512, 256)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin6',nn.Linear(256, 64)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin7',nn.Linear(64, output_size))]
                           ))
 
    def target_forward(target):
        tembeds = self.target_embeddings(target).view((bs, -1))
        return self.target_position_network(tembeds) 
          
    def forward(self, angle, target_out):
        bs = angle.shape[0]
        embeds = self.embeddings(angle).view((bs, -1))
        aout = self.angle_network(embeds) 
        # angle outptu 
        return self.output_angle_network(torch.cat((target_out,aout),1))

class JumboAngleAutoEncoder(nn.Module):
    def __init__(self, input_size=5, output_size=1):
        super(JumboAngleAutoEncoder, self).__init__()
        # bs,1,1,states
        self.angle_network = nn.Sequential(
                         OrderedDict([
                            ('lin0',nn.Linear(input_size, 128)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin1',nn.Linear(128, 256)),
                            ('relu',nn.ReLU(True)), 
                            ('lin2',nn.Linear(256, 512)),
                            ('relu',nn.ReLU(True)), 
                            ('lin3',nn.Linear(512, 1024)),
                            ('relu',nn.ReLU(True)), 
                            ('lin4',nn.Linear(1024, 1024)),
                            ]))
                           
        self.target_position_network = nn.Sequential(
                         OrderedDict([
                            ('lin0',nn.Linear(3, 128)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin1',nn.Linear(128, 256)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin2',nn.Linear(256, 512)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin3',nn.Linear(512, 1024)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin4',nn.Linear(1024, 1024)), 
                            ]))
 
        self.output_angle_network = nn.Sequential(
                         OrderedDict([
                            ('lin0',nn.Linear(1024*2, 1024)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin1',nn.Linear(1024, 512)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin2',nn.Linear(512, 128)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin3',nn.Linear(128, 64)), 
                            ('relu',nn.ReLU(True)), 
                            ('lin4',nn.Linear(64, output_size))]
                           ))
 
    def forward(self, angle, target):
        aout = self.angle_network(angle) 
        tout = self.target_position_network(target) 
        # angle outptu 
        return self.output_angle_network(torch.cat((tout,aout),1))
