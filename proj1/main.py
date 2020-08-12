#coding:utf-8
import numpy as np
from spclustering import Spectrum

data= np.load('D:/action-sets-master/action-sets-master/data/features/' + 'P03_cam01_P03_cereals.npy').T
sp = Spectrum(n_cluster=5, method='normalized', criterion='k_nearest', gamma=0.1,k=0)
sp.fit(data)

label2index = dict()
index2label = dict()
with open('D:/action-sets-master/action-sets-master/data/mapping.txt', 'r') as f:
     content = f.read().split('\n')[0:-1]
     for line in content:
            label2index[line.split()[1]] = int(line.split()[0])
            index2label[int(line.split()[0])] = line.split()[1]
with open('D:/action-sets-master/action-sets-master/data/groundTruth/' + 'P03_cam01_P03_cereals.txt') as f:
     ground_truth = [ label2index[line] for line in f.read().split('\n')[0:-1] ]
      
print(sp.labels)
print(np.array(ground_truth))
