import torch
import torch.nn as nn

import torch.utils.data as data


torch.manual_seed(1)

use_gpu = torch.cuda.is_available()
device = torch.device('cuda:0' if use_gpu else 'cpu')


class RNNDataset(data.Dataset):

    def __init__(self, data, labels, future):
        self.data = data
        self.labels = labels
        self.future = future

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        features = self.data[index]
        label = self.labels[index]
        future = self.future[index]

        return features, label, future


class StatelessRNN(nn.Module):

    def __init__(self, input_dim, hidden_size):
        super(StatelessRNN, self).__init__()

        self.input_size = input_dim
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(self.input_size, self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size + 20, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, future):

        x = x.permute(1, 0, 2)

        batch_size = x[0].size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_size).float()
        h1 = torch.zeros(1, batch_size, self.hidden_size).float()

        h0, h1 = h0.to(device), h1.to(device)
        hidden = (h0, h1)

        output, hidden = self.lstm(x.float(), hidden)

        x = torch.cat((output[-1].float(), future.float()), dim=1)

        result = self.dropout(self.relu(self.fc1(x)))
        result = self.fc2(result)

        return result
