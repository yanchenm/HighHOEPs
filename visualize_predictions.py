'''
this is a file to visualize model outputs
'''
from train import *
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
import time

model_type = 'rnn'
model = torch.load('./outputs/model_lstm_3layer.pt')
model1 = torch.load('./outputs/model_lstm_3layer_L1.pt')


a = pd.read_csv('./data/output/final_data.csv', header=0, parse_dates=[0])

#user_input = input("\nEnter a date between 2018-11-14 and 2018-11-11\n")

#hour = pd.to_datetime(user_input + ' 0:00')
hour = pd.to_datetime('2018-11-15 6:00')

a.iloc[:, 1:] = a.iloc[:, 1:].astype(float)

mean = a.iloc[:, 19:].mean(0)
std = a.iloc[:, 19:].std(0)
a.iloc[:, 19:] = (a.iloc[:, 19:] - mean) / std


d = {}
l = {}
while hour < pd.to_datetime('2018-11-24 16:00'):
    data, labels = get_train_instance(hour, a, model_type)
    # print(hour)
    if type(data) != int:
        data = torch.tensor(data).float()
        d[hour] = data
        l[hour] = labels
    hour = hour + pd.Timedelta(hours=1)

predictions = []
big_graph=[]
for datehour in d:
    p = model(d[datehour]).detach().numpy().astype(float)
    p1 = model1(d[datehour]).detach().numpy().astype(float)

    predictions.append([p,p1])
    big_graph.append([datehour,p[0],l[datehour]])
    print(datehour,p,p1,l[datehour])

pred = [[item[0].squeeze() for item in predictions]]
pred = np.array(pred)
big_graph = np.array(big_graph)
fig = plt.figure(figsize=(15,5))
s = int(a.timestamp[a.timestamp == pd.to_datetime('2018-11-15 6:00')].index[0])
graph_data = a.iloc[s:]
x_axis = graph_data['timestamp'].dt.month.astype(str) + "-" + graph_data['timestamp'].dt.day.astype(str) + " " + \
         graph_data['timestamp'].dt.hour.astype(str) + ":00"
plt.plot(big_graph[:,0],big_graph[:,1],linestyle = '--',dashes = (1,1),label = 'model predictions')
plt.plot(big_graph[:,0],big_graph[:,2],label = 'HOEP')
plt.xticks([])
plt.legend(loc=1)
plt.ylabel('price ($/MWh)')
plt.show()

i = 0
def onclick(fig):
    fig.clear()
    global i
    s = int(a.timestamp[a.timestamp == list(d)[i]].index[0])
    graph_data = a.iloc[s-5:s+5]
    pred = predictions[i][0] #+ graph_data['price_pd_3'].iloc[7]
    pred1 = predictions[i][1]

    x_axis = graph_data['timestamp'].dt.month.astype(str) + "-" + graph_data['timestamp'].dt.day.astype(str) + " "+graph_data['timestamp'].dt.hour.astype(str)+":00"
    plt.plot(x_axis,graph_data['hoep'],color = 'b',label = 'HOEP = {}'.format(graph_data['hoep'].iloc[7]))
    plt.plot(x_axis,graph_data['price_pd_3'],linestyle = '--',dashes = (2,5),color = 'b', label = 'PD-3 Price = {}'.format(graph_data['price_pd_3'].iloc[7]))
    plt.plot(x_axis.iloc[7],pred,'ro',label = 'pred = {}'.format('%.2f' % pred[0,0]))
    plt.plot(x_axis.iloc[7],pred1,'ro',label = 'pred1 = {}'.format('%.2f' % pred1[0,0]))
    plt.axvline(x = x_axis.iloc[5])

    plt.legend(loc=1)
    plt.xticks(rotation=20)
    plt.ylim(-5,80)
    plt.draw()
    i += 1

fig = plt.figure(figsize=(8,5))
#fig.canvas.mpl_connect('button_press_event', lambda event: onclick(fig))
#plt.show()
while True:
    onclick(fig)
    plt.pause(0.2)



