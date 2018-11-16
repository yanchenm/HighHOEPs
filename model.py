'''
This file contains the pytorch model(s) used in the HighHOEPs project

Last edit: 2018-11-11 18:00
Editor: Sam Harrison
'''

import torch.nn as nn
import torch.nn.functional as F



class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(332,10)
        #self.fc2 = nn.Linear(500,200)
        self.fc3 = nn.Linear(10,1)

    def forward(self, x):
        #print(x)
        #print(self.fc1.weight)
        x = self.fc1(x)
        #print('1',x)
        #x = self.fc2(x)
        #print('2',x)
        x = self.fc3(x)
        #print('3',x)
        return x

