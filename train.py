'''
This file contains the pytorch model(s) used in the HighHOEPs project

Last edit: 2018-11-11 18:30
Editor: Sam Harrison
'''

import torch
from model import *
import pandas as pd
import numpy as np
from linear_dataset import *
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def train_model(train_loader, validation_loader, model, loss_fnc, optmizer, epochs):
    #trans the model "model" for epochs using optmizer and loss_fnc
    #validation iterator used once/epoch
    #returns trained model

    graph_data = pd.DataFrame(columns=['epoch','training loss','validation loss'])

    for epoch in range(epochs):
        accum_loss = 0
        batch_count = 0

        for i, batch in enumerate(train_loader):
            data, labels = batch
            optmizer.zero_grad()
            predictions = model(data.float())
            batch_loss = loss_fnc(input=predictions,target=labels.float())
            accum_loss += batch_loss
            batch_loss.backward()
            optmizer.step()
            batch_count = i

        accum_loss = accum_loss/batch_count
        validation_loss = evaluate_model(validation_loader,model,loss_fnc)
        print("Epoch:{} | training loss:{} | validation loss:{}"
              .format(epoch+1,'%.2f' % accum_loss,'%.2f' % validation_loss))
        graph_data = graph_data.append({'epoch':epoch,'training loss':accum_loss,'validation loss':validation_loss}, ignore_index=True)
    print("\n\n")
    return model, graph_data

def evaluate_model(validation_loader,model,loss_fnc):
    #evaluate model
    #returns error rate and per-sample loss

    accum_loss = 0
    batch_count = 0

    for i,batch in enumerate(validation_loader):
        data, labels = batch
        predictions = model(data.float())

        batch_loss = loss_fnc(input=predictions, target=labels.float())
        accum_loss += batch_loss
        batch_count = i

    return accum_loss/batch_count

def get_validation_residuals(validation_loader,model,model_type):
    '''
    This function returns a dataframe with three columns, HOEP, PD-3 Price, and model prediction
    validation load is a DataLoader type containing validation set
    model is model used to make the predictions
    model type must be "linear", "sig", or "rnn"
    '''
    res_graph = pd.DataFrame(columns=['hoep', 'PD-3', 'model_pred'])

    for i, batch in enumerate(validation_loader):
        data, labels = batch
        if model_type == "rnn":
            PD_3 = data[:,7,16]
        else:
            PD_3 = data[:, 308]

        predictions = model(data.float())
        a = np.array([labels.squeeze().detach().numpy(), PD_3.squeeze().detach().numpy(),
                      predictions.squeeze().detach().numpy()])
        a = np.swapaxes(a, 0, 1)
        df = pd.DataFrame(a, columns=['hoep', 'PD-3', 'model_pred'])
        res_graph = res_graph.append(df)

    return res_graph

def load_model(lr,type):    #load linear model
    loss_fnc = torch.nn.MSELoss()
    if type == 'linear':
        model = LinearModel()
    if type == 'sig':
        model = LinearSigModel()
    if type == 'rnn':
        model = RNN()
    optimizer = torch.optim.Adam(model.parameters(),lr)

    return model, loss_fnc, optimizer

def get_train_instance(hour,data_array,model_type):
    # returns a 1x332 array of values to be fed into linear NN and a 1x5 array of prices

    if True not in (data_array.timestamp == hour).values:
        return 0,0
    s = int(data_array.timestamp[data_array.timestamp == hour].index[0])

    a = data_array.iloc[s-5:s+5]
    a = a.values

    outline = np.genfromtxt('train_data_shape.csv', delimiter=",",skip_header=True)
    outline = np.flipud(outline)

    if model_type == "sig" or model_type=="linear":
        data = a[outline == 1]
        data = data.flatten()
        data = data.astype(float)
        labels = np.array([a[7,1]])
        labels = labels.astype(float)

    if model_type == "rnn":
        outline = outline[6:,:]
        labels = np.array([a[7, 1]])
        labels = labels.astype(float)
        data = a
        data[6:,:][outline != 1] = 0
        data = data[:,1:]
        data = data.astype(float)
        data = np.expand_dims(data,0)

    return data,labels

def load_data(batch_size,dataset,labelset):

    data_train, data_val, labels_train, labels_val = train_test_split(dataset, labelset, test_size=0.3,
                                                                      random_state=1)
    #data_train, data_val = dataset[240:], dataset[:240]
    #labels_train, labels_val = labelset[240:], labelset[:240]

    training_dataset = LinearDataset(data_train, labels_train)
    validation_dataset = LinearDataset(data_val, labels_val)

    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)


    return train_loader, val_loader

if __name__ == "__main__":

    '''
    define model type and filename here.
    model_type must be "linear", "sig", or "rnn"
    save is True means code will save model, training loss graph, and scatter plot to working directory
    model_name is appended to the name of saved files
    '''
    model_type = 'rnn'
    model_name = 'lstm'
    save = True


    a = pd.read_csv('final_data.csv',header=0,parse_dates=[0])          #take data input
    hour = pd.to_datetime('2018-10-13 5:00')
    a.iloc[:,1:] = a.iloc[:,1:].astype(float)

    mean = a.iloc[:,19:].mean(0)
    std = a.iloc[:,19:].std(0)
    a.iloc[:,19:] = (a.iloc[:,19:] - mean) / std                # normalize


    if model_type == 'rnn': dataset = np.ndarray((0,10,50))
    else: dataset = np.ndarray((0,332))

    labelset = np.ndarray((0,1))

    while hour < pd.to_datetime('2018-11-12 17:00'):
        data, labels = get_train_instance(hour, a, model_type)
        hour = hour + pd.Timedelta(hours=1)
        #print(hour)
        if type(data) != int:
            dataset = np.vstack((dataset,data))
            labelset = np.vstack((labelset,labels))


    print('dataset created')

    batch_size, lr, epochs = 5, 0.00001, 100

    train_loader, val_loader = load_data(batch_size,dataset,labelset)
    print('dataloaders created')

    Linear, loss_fnc, optimizer = load_model(lr,model_type)

    '''
    This section trains the model, saves the trained model, and creates a plot of training and validation loss
    '''
    Linear, graph_data = train_model(train_loader,val_loader,Linear,loss_fnc,optimizer,epochs)
    if save: torch.save(Linear, 'model_{}.pt'.format(model_name))
    plt.figure(1,)
    plt.plot(graph_data['epoch'],graph_data['training loss'],label = 'training loss')
    plt.plot(graph_data['epoch'],graph_data['validation loss'],label = 'validation loss = {}'.format(graph_data['validation loss'].iloc[epochs-1]))
    plt.legend(loc='best')
    if save: plt.savefig('plot_training_{}.png'.format(model_name))


    '''
    This section creates two scatter plots 
    -HOEP vs PD-3 price
    -HOEP vs model predictions
    '''
    res_graph = get_validation_residuals(val_loader,Linear,model_type)
    plt.figure(2,figsize=(10,5))
    plt.subplot(1,2,1)
    sse = ((res_graph['hoep']-res_graph['PD-3'])**2).sum()
    plt.scatter(res_graph['hoep'],res_graph['PD-3'],s=5,label = "sse = {}".format(sse))
    plt.xlabel('hoep')
    plt.ylabel('ieso PD-3 price')
    plt.ylim((0,200))
    plt.xlim((0,200))
    plt.legend(loc='best')
    plt.subplot(1, 2, 2)
    sse = ((res_graph['hoep']-res_graph['model_pred'])**2).sum()
    plt.scatter(res_graph['hoep'], res_graph['model_pred'],s=5,label = "sse = {}".format(sse))
    plt.ylabel('model prediction')
    plt.xlabel('hoep')
    plt.ylim((0,200))
    plt.xlim((0,200))
    plt.legend(loc='best')
    if save: plt.savefig('plot_residuals_{}.png'.format(model_name))
    plt.show()







