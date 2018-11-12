'''
This file contains the pytorch model(s) used in the HighHOEPs project

Last edit: 2018-11-11 18:00
Editor: Sam Harrison
'''

import torch.nn as nn
import torch.nn.functional as F



class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.fc1 = nn.Linear(332,500)
        self.fc2 = nn.Linear(500,200)
        self.fc3 = nn.Linear(200,5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

