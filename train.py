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

def train_model(train_iterator, validation_iterator, model, loss_fnc, optmizer, epochs):
    #trans the model "model" for epochs using optmizer and loss_fnc
    #validation iterator used once/epoch
    #returns trained model

    for epoch in range(epochs):
        i=0
        accum_correct = 0
        accum_loss = 0

        for batch in train_iterator:

            if i > len(train_iterator.dataset.examples)/batch.batch_size: break
            i += 1
            labels = batch.label
            sentences = batch.text
            optmizer.zero_grad()
            predictions = model(sentences[0],lengths=sentences[1])
            batch_loss = loss_fnc(input=predictions,target=labels.float())
            accum_loss += batch_loss
            batch_loss.backward()
            optmizer.step()

            correct = (predictions > 0.5).squeeze().long() == labels
            accum_correct += int(correct.sum())

        accum_loss = accum_loss/len(train_iterator.dataset.examples)*1000
        training_error = 1 - accum_correct/len(train_iterator.dataset.examples)
        validation_error, validation_loss = evaluate_model(validation_iterator,model,loss_fnc)
        print("Epoch:{} | training error rate:{} | validation error rate:{} | training loss:{} | validation loss:{}"
              .format(epoch+1,training_error,validation_error,accum_loss,validation_loss))
    print("\n\n")
    return model

def evaluate_model(validation_iterator,model,loss_fnc):
    #evaluate model
    #returns error rate and per-sample loss
    i = 0
    accum_correct = 0
    accum_loss = 0
    for batch in validation_iterator:
        if i > len(validation_iterator.dataset.examples) / batch.batch_size: break
        i += 1

        labels = batch.label
        sentences = batch.text
        predictions = model(sentences[0], lengths=sentences[1])
        batch_loss = loss_fnc(input=predictions, target=labels.float())
        accum_loss += batch_loss
        correct = (predictions > 0.5).squeeze().long() == labels
        accum_correct += int(correct.sum())

    return 1 - accum_correct/len(validation_iterator.dataset.examples), accum_loss/len(validation_iterator.dataset.examples)*1000

def load_models(lr):    #load linear model
    loss_fnc = torch.nn.BCELoss()
    Linear = Linear()
    optimizer = torch.optim.SGD(Linear.parameters(),lr)

    return Linear, loss_fnc, optimizer

def get_train_instance(hour,data_array):
    # returns a 1x332 array of values to be fed into linear NN and a 1x5 array of prices

    s = int(data_array.DateHour[data_array.DateHour == hour].index[0])

    a = data_array.iloc[s-4:s+6]
    a = a.values

    outline = np.genfromtxt('train_data_shape.csv', delimiter=",",skip_header=True)
    #a = np.zeros_like(outline) #a is subbing in for a 10*51 array from the database

    data = a[outline == 1]
    data = data.flatten()

    labels = a[2,1]

    return data,labels

if __name__ == "__main__":
    a = pd.read_csv('draft_db_fields.csv',header=0,parse_dates=[0]) #this is example of data
    hour = pd.to_datetime('2018-11-11 15:00')
    data, labels = get_train_instance(hour,a)
    print(data,data.shape,labels)