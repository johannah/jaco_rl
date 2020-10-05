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
