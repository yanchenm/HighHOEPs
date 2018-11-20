'''
This file contains the pytorch model(s) used in the HighHOEPs project

Last edit: 2018-11-11 18:00
Editor: Sam Harrison
'''

import torch.nn as nn
import torch.nn.functional as F
import torch



class LinearSigModel(nn.Module):
    def __init__(self):
        super(LinearSigModel, self).__init__()
        self.fc1 = nn.Linear(332,500)
        self.fc1_sig = nn.Linear(332,250)
        self.fc2 = nn.Linear(500,200)
        self.fc2_sig = nn.Linear(250,100)
        self.fc4 = nn.Linear(300,1)

    def forward(self, x):

        x1 = F.relu(self.fc1(x))
        x2 = F.sigmoid(self.fc1_sig(x))
        if len(x1.shape) == 1:
            x1 = x1.unsqueeze(0)
            x2 = x2.unsqueeze(0)

        x1 = F.relu(self.fc2(x1))
        x2 = F.sigmoid(self.fc2_sig(x2))
        x = torch.cat((x1,x2),1)
        x = self.fc4(x)
        return x

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(332,500)
        self.fc2 = nn.Linear(500,200)
        self.fc3 = nn.Linear(200,1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(50,100,1)
        self.fc1 = nn.Linear(400, 200)
        self.fc2 = nn.Linear(200, 1)

    def forward(self, x):
        #x1 is past hours
        #x2 is PD info for next four hours
        x1 = x[:,:6,:]
        x2 = x[:,6:,:]
        batch_size = x1.shape[0]
        x1 = x1.permute(1,0,2)
        hidden_out, (x1a,x1b) = self.lstm(x1)
        x1 = torch.cat((x1a,x1b),2)
        x2 = x2.view(batch_size,4*50)
        x1 = x1.view(batch_size,200)
        x = torch.cat((x1,x2),1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

