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
        accum_correct = 0
        accum_loss = 0

        for i, batch in enumerate(train_loader):
            data, labels = batch
            optmizer.zero_grad()
            predictions = model(data.float())
            #print(predictions)
            batch_loss = loss_fnc(input=predictions,target=labels.float())
            if i == 10: print(predictions,labels)
            accum_loss += batch_loss
            batch_loss.backward()
            optmizer.step()
            #correct = (predictions > 0.5).squeeze().long() == labels
            #accum_correct += int(correct.sum())

        #print(accum_loss,j)
        accum_loss = accum_loss/train_loader.dataset.data.shape[0]
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
    for i,batch in enumerate(validation_loader):
        data, labels = batch

        predictions = model(data.float())

        batch_loss = loss_fnc(input=predictions, target=labels.float())
        accum_loss += batch_loss

        #correct = (predictions > 0.5).squeeze().long() == labels
        #accum_correct += int(correct.sum())

    return accum_loss/validation_loader.dataset.data.shape[0]

def load_model(lr):    #load linear model
    loss_fnc = torch.nn.MSELoss()
    Linear = LinearModel()
    optimizer = torch.optim.SGD(Linear.parameters(),lr)

    return Linear, loss_fnc, optimizer

def get_train_instance(hour,data_array):
    # returns a 1x332 array of values to be fed into linear NN and a 1x5 array of prices

    if True not in (data_array.timestamp == hour).values:
        return 0,0
    s = int(data_array.timestamp[data_array.timestamp == hour].index[0])

    a = data_array.iloc[s-4:s+6]
    a = a.values
    a[:,19:] = a[:,19:]/1000

    outline = np.genfromtxt('train_data_shape.csv', delimiter=",",skip_header=True)
    #a = np.zeros_like(outline) #a is subbing in for a 10*51 array from the database

    data = a[outline == 1]
    data = data.flatten()

    labels = np.array([a[2,1]])
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
    hour = pd.to_datetime('2018-10-13 4:00')
    dataset = []
    labelset = []
    while hour < pd.to_datetime('2018-11-12 17:00'):
        data, labels = get_train_instance(hour, a)
        hour = hour + pd.Timedelta(hours=1)
        #print(hour)
        if type(data) != int:
            dataset.append(data)
            labelset.append(labels)
    dataset = np.array(dataset)
    labelset = np.array(labelset)#.squeeze()
    dataset = dataset.astype(float)
    labelset = labelset.astype(float)

    print('dataset created')

    batch_size, lr, epochs = 25, 0.01, 50

    train_loader, val_loader = load_data(batch_size,dataset,labelset)
    print('dataloaders created')

    Linear, loss_fnc, optimizer = load_model(lr)

    model, graph_data = train_model(train_loader,val_loader,Linear,loss_fnc,optimizer,epochs)

    plt.figure()
    plt.plot(graph_data['epoch'],graph_data['training loss'],label = 'training loss')
    plt.plot(graph_data['epoch'],graph_data['validation loss'],label = 'validation loss')
    plt.legend(loc='best')
    plt.show()




