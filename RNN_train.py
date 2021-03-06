import os
import argparse

from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

from RNN import *
from RNN_utils import *


matplotlib.use('TkAgg')

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)


def evaluate(model, val_loader, loss_fnc):
    accum_loss = 0.0
    batch_count = 0

    model.eval()

    for i, batch in enumerate(val_loader):
        features, labels, future = batch
        features, labels, future = features.to(device), labels.to(device), future.to(device)

        predictions = model(features, future)
        batch_loss = loss_fnc(input=predictions.squeeze(), target=labels.float())

        accum_loss += batch_loss.item()
        batch_count = i

        del predictions
        del batch_loss

    val_loss = accum_loss / batch_count

    return val_loss


def load_data(features, labels, future, batch_size):

    train_feats, val_feats, train_labels, val_labels, train_future, val_future = train_test_split(features, labels,
                                                                                                  future, test_size=0.2,
                                                                                                  shuffle=False)

    train_dataset = RNNDataset(train_feats, train_labels, train_future)
    val_dataset = RNNDataset(val_feats, val_labels, val_future)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def load_model(type, input_dim, hidden_dim, lr):
    if type == 'stateful':
        model = StatefulRNN(input_dim, hidden_dim)
    else:
        model = StatelessRNN(input_dim, hidden_dim)

    model.to(device)

    loss_fnc = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

    return model, loss_fnc, optimizer


def plot_predictions(model_path, data_path):

    model = torch.load(model_path)
    model.to(device)
    model.eval()

    test_data = pd.read_csv(data_path)
    np_data = test_data[test_data.columns[2:51]].values
    labels = test_data['hoep'].values
    price_pd = test_data['price_pd_5'].tail(len(test_data) - 49).values

    timestamps = test_data['timestamp']
    timestamps = timestamps.apply(datetime.strptime, args=('%Y-%m-%d %H:%M:%S',))
    timestamps = timestamps.tail(len(timestamps) - 49)

    features, labels, future = window_subsample(test_data[test_data.columns[2:51]], labels, 50)
    features = torch.Tensor(features).to(device)
    future = torch.Tensor(future).to(device)

    predictions = model(features, future).squeeze()
    predictions = predictions.cpu().detach().numpy()

    print(predictions)

    plt.plot(np.arange(0, len(labels), 1), labels, 'b-', predictions, 'r--', price_pd, 'k--')
    plt.show()


def scatter_plot(model_path, data_path):

    model = torch.load(model_path)
    model.to(device)
    model.eval()

    data = pd.read_csv(data_path)
    labels = data['hoep'].values

    pd_3 = data['price_pd_3'].values
    pd_3 = pd_3[50:]

    features, labels, future = window_subsample(data[data.columns[2:51]], labels, 50)

    train_feats, val_feats, train_labels, \
    val_labels, train_future, val_future, train_pd, val_pd = train_test_split(features, labels, future, pd_3,
                                                                              test_size=0.3, random_state=1)

    features = torch.Tensor(val_feats).to(device)
    future = torch.Tensor(val_future).to(device)

    predictions = model(features, future).squeeze()
    predictions = predictions.cpu().detach().numpy()

    res_graph = pd.DataFrame({'hoep': val_labels, 'PD-3': val_pd, 'model_pred': predictions})

    plt.figure(2, figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sse = ((res_graph['hoep'] - res_graph['PD-3']) ** 2).sum()
    plt.scatter(res_graph['hoep'], res_graph['PD-3'], s=5, label="sse = {}".format(sse))
    plt.xlabel('hoep')
    plt.ylabel('ieso PD-3 price')
    plt.ylim((0, 200))
    plt.xlim((0, 200))
    plt.legend(loc='best')
    plt.subplot(1, 2, 2)
    sse = ((res_graph['hoep'] - res_graph['model_pred']) ** 2).sum()
    plt.scatter(res_graph['hoep'], res_graph['model_pred'], s=5, label="sse = {}".format(sse))
    plt.ylabel('model prediction')
    plt.xlabel('hoep')
    plt.ylim((0, 200))
    plt.xlim((0, 200))
    plt.legend(loc='best')
    plt.savefig('./fig/plot_residuals_{}.png'.format('stateless_rnn'))
    plt.show()


def train_stateless(args):

    # Extract arguments
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    type = args.type

    # Load data
    data = pd.read_csv('./data/output/normalized_data.csv')
    labels = data['hoep'].values

    features, labels, future = window_subsample(data[data.columns[2:51]], labels, 50)

    train_loader, val_loader = load_data(features, labels, future, batch_size)
    model, loss_fnc, optimizer = load_model(type, 49, 300, lr)

    # Training performance tracking
    performance = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss'])

    min_val_loss = 100000

    for epoch in range(epochs):
        accum_loss = 0.0
        batch_count = 0
        model.train()

        if epoch == 50 or epoch == 100 or epoch == 150:
            lr = lr / 10

            for group in optimizer.param_groups:
                group['lr'] = lr

        for i, batch in enumerate(train_loader):
            features, labels, future = batch
            features, labels, future = features.to(device), labels.to(device), future.to(device)

            optimizer.zero_grad()

            predictions = model(features, future)
            batch_loss = loss_fnc(input=predictions.squeeze(), target=labels.float())

            batch_loss.backward()
            optimizer.step()

            accum_loss += batch_loss.item()
            batch_count = i

            del batch_loss
            del predictions

        train_loss = accum_loss / batch_count
        val_loss = evaluate(model, val_loader, loss_fnc)

        performance = performance.append({'epoch': epoch, 'train_loss': accum_loss, 'val_loss': val_loss},
                                         ignore_index=True)

        print("Epoch: {} | training loss: {:.4f} | validation loss: {:.4f}".format(epoch + 1, train_loss, val_loss))

        if val_loss < min_val_loss:
            torch.save(model, './models/model_stateless.pt')

    performance.to_csv('./data/output/train_performance.csv', index=False)

    plot_predictions('./models/model_stateless.pt', './data/test/output/normalized_data.csv')
    scatter_plot('./models/model_stateless.pt', './data/output/normalized_data.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--type', type=str, default='stateless',
                        help="RNN type: stateless or stateful")

    args = parser.parse_args()

    train_stateless(args)
