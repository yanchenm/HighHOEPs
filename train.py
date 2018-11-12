'''
This file contains the pytorch model(s) used in the HighHOEPs project

Last edit: 2018-11-11 18:30
Editor: Sam Harrison
'''

import torch
from model import *

def train_model(train_iterator, validation_iterator, model, loss_fnc, optmizer, epochs):
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

def load_models(lr):
    loss_fnc = torch.nn.BCELoss()
    Linear = Linear()
    optimizer = torch.optim.SGD(Linear.parameters(),lr)

    return Linear, loss_fnc, optimizer