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
        self.fc1 = nn.Linear(self.hidden_size + 20, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(32, 1)

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


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_size, steps):
        super(encoder, self).__init__()

        self.input_size = input_dim
        self.hidden_size = hidden_size
        self.steps = steps

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, 1)
        self.attention = nn.Linear(2 * self.hidden_size + self.steps - 1, 1)

    def forward(self, x):

        batch_size = x.size(0)

        input_weighted = torch.zeros(batch_size, self.steps - 1, self.input_size)
        input_encoded = torch.zeros(batch_size, self.steps - 1, self.input_size)

        h0 = torch.zeros(1, batch_size, self.hidden_size).float()
        h1 = torch.zeros(1, batch_size, self.hidden_size).float()

        input_weighted, input_encoded = input_weighted.to(device), input_encoded.to(device)
        h0, h1 = h0.to(device), h1.to(device)

        for t in range(self.steps):
            # Concatenate the previous hidden state with the current input
            data = torch.cat((h0.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                              h1.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                              x.permute(0, 2, 1)), dim=2)

            # Calculate attention weights with linear network
            data = self.attention(data.view(-1, self.hidden_size * 2 + self.steps - 1))
            attention_weights = nn.Softmax(data.view(-1, self.input_size), dim=1)

            # Weight the input using new attention weights
            x_tilde = torch.mul(attention_weights, x[:, t, :])

            # Update the hidden state
            _, (h0, h1) = self.lstm(x_tilde, (h0, h1))

            input_weighted[:, t, :] = x_tilde
            input_encoded[:, t, :] = h0

        return input_weighted, input_encoded


class Decoder(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, steps):
        super(decoder, self).__init__()

        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.attention = nn.Linear(2 * self.decoder_hidden_size + self.encoder_hidden_size, encoder_hidden_size)
        self.tanh = nn.Tanh()

    def forward(self):
        pass
