'''
This file contains the pytorch model(s) used in the HighHOEPs project

Last edit: 2018-11-11 18:30
Editor: Sam Harrison
'''

import torch
from model import *
import pandas as pd
import numpy as np
import datetime as dt
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
            #if i == 10: print(predictions,labels)
            accum_loss += batch_loss
            batch_loss.backward()
            optmizer.step()
            batch_count = i

        #print(accum_loss,j)
        len = train_loader.dataset.data.shape[0]
        accum_loss = accum_loss/batch_count #/len
        #training_error = 1 - accum_correct/len(train_loader.dataset.examples)
        validation_loss = evaluate_model(validation_loader,model,loss_fnc)
        print("Epoch:{} | training loss:{} | validation loss:{}"
              .format(epoch+1,accum_loss,validation_loss))
        graph_data = graph_data.append({'epoch':epoch,'training loss':accum_loss,'validation loss':validation_loss}, ignore_index=True)
    print("\n\n")
    return model, graph_data

def evaluate_model(validation_loader,model,loss_fnc):
    #evaluate model
    #returns error rate and per-sample loss

    accum_correct = 0
    accum_loss = 0
    batch_count = 0
    for i,batch in enumerate(validation_loader):
        data, labels = batch

        predictions = model(data.float())

        batch_loss = loss_fnc(input=predictions, target=labels.float())
        accum_loss += batch_loss
        batch_count = i

    len = validation_loader.dataset.data.shape[0]
    return accum_loss/batch_count

def load_model(lr):    #load linear model
    loss_fnc = torch.nn.MSELoss()
    Linear = LinearModel()
    optimizer = torch.optim.Adam(Linear.parameters(),lr)

    return Linear, loss_fnc, optimizer

def get_train_instance(hour,data_array):
    # returns a 1x332 array of values to be fed into linear NN and a 1x5 array of prices

    if True not in (data_array.timestamp == hour).values:
        return 0,0
    s = int(data_array.timestamp[data_array.timestamp == hour].index[0])

    a = data_array.iloc[s-5:s+5]
    a = a.values


    outline = np.genfromtxt('train_data_shape.csv', delimiter=",",skip_header=True)
    #a = np.zeros_like(outline) #a is subbing in for a 10*51 array from the database
    outline = np.flipud(outline)
    data = a[outline == 1]
    data = data.flatten()
    data = data.astype(float)

    labels = np.array([a[7,1]])
    labels = labels.astype(float)
    #labels = np.array([a[:5,1]])

    return data,labels

def load_data(batch_size,dataset,labelset):

    data_train, data_val, labels_train, labels_val = train_test_split(dataset, labelset, test_size=0.3,
                                                                      random_state=0)

    training_dataset = LinearDataset(data_train, labels_train)
    validation_dataset = LinearDataset(data_val, labels_val)

    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)


    return train_loader, val_loader

if __name__ == "__main__":
    a = pd.read_csv('final_data.csv',header=0,parse_dates=[0]) #this is example of data
    hour = pd.to_datetime('2018-10-13 5:00')
    a.iloc[:,1:] = a.iloc[:,1:].astype(float)

    mean = a.iloc[:,19:].mean(0)
    std = a.iloc[:,19:].std(0)
    a.iloc[:,19:] = (a.iloc[:,19:] - mean) / std


    dataset = np.ndarray((0,332))
    labelset = np.ndarray((0,1))
    while hour < pd.to_datetime('2018-11-12 17:00'):
        data, labels = get_train_instance(hour, a)
        hour = hour + pd.Timedelta(hours=1)
        #print(hour)
        if type(data) != int:
            dataset = np.vstack((dataset,data))
            labelset = np.vstack((labelset,labels))


    print('dataset created')

    batch_size, lr, epochs = 15, 0.00001, 100

    train_loader, val_loader = load_data(batch_size,dataset,labelset)
    print('dataloaders created')

    Linear, loss_fnc, optimizer = load_model(lr)

    model, graph_data = train_model(train_loader,val_loader,Linear,loss_fnc,optimizer,epochs)
    torch.save(model, 'linear_model.pt')
    plt.figure()
    plt.plot(graph_data['epoch'],graph_data['training loss'],label = 'training loss')
    plt.plot(graph_data['epoch'],graph_data['validation loss'],label = 'validation loss')
    plt.legend(loc='best')
    plt.show()






