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

model = torch.load('linear_model.pt')

a = pd.read_csv('final_data.csv', header=0, parse_dates=[0])  # this is example of data
hour = pd.to_datetime('2018-10-13 4:00')
a.iloc[:, 1:] = a.iloc[:, 1:].astype(float)

mean = a.iloc[:, 19:].mean(0)
std = a.iloc[:, 19:].std(0)
a.iloc[:, 19:] = (a.iloc[:, 19:] - mean) / std

d = {}
l = {}
while hour < pd.to_datetime('2018-11-12 17:00'):
    data, labels = get_train_instance(hour, a)
    # print(hour)
    if type(data) != int:
        #ata = np.expand_dims(data,0)
        data = torch.tensor(data).float()
        d[hour] = data
        l[hour] = labels
        #dataset = np.vstack((dataset, data))
        #labelset = np.vstack((labelset, labels))
    hour = hour + pd.Timedelta(hours=1)

predictions = []
for datehour in d:
    p = float(model(d[datehour]))
    predictions.append(p)
    print(datehour,p,l[datehour])

i = 0
def onclick1(fig):
    fig.clear()
    global i
    s = int(a.timestamp[a.timestamp == list(d)[i]].index[0])
    graph_data = a.iloc[s-4:s+6]
    x_axis = graph_data['timestamp'].dt.month.astype(str) + "-" + graph_data['timestamp'].dt.day.astype(str) + " "+graph_data['timestamp'].dt.hour.astype(str)
    plt.plot(x_axis,graph_data['hoep'],color = 'b')
    plt.plot(x_axis,graph_data['price_pd_3'],linestyle = '--',dashes = (5,5),color = 'b')
    plt.plot(x_axis.iloc[-3],predictions[i],'ro',label = 'pred = {}'.format(predictions[i]))
    plt.legend(loc='best')
    plt.xticks(rotation=45)
    plt.ylim(-5,80)
    plt.draw()
    i += 1

fig = plt.figure()
fig.canvas.mpl_connect('button_press_event', lambda event: onclick1(fig))

plt.show()
