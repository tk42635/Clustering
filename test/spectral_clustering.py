#coding:utf-8
import numpy as np
from sklearn.cluster import KMeans


class Spectrum:
    def __init__(self, n_cluster, method, criterion, gamma=2, dis_epsilon=70, k=400,tol=1e-3, max_iter=1000,):
        self.n_cluster = n_cluster
        self.tol = tol
        self.max_iter = max_iter
        self.method = method  
        self.criterion = criterion  
        self.gamma = gamma  
        self.dis_epsilon = dis_epsilon  
        self.k = k  

        self.W = None  
        self.L = None  
        self.L_norm = None  
        self.D = None  
        self.cluster = None

        self.N = None

    def init_param(self, data):
        self.N = data.shape[0]
        dis_mat = self.cal_dis_mat(data)
        self.cal_weight_mat(dis_mat)
        self.D = np.diag(self.W.sum(axis=1))
        self.L = self.D - self.W
        return

    def cal_dis_mat(self, data):
        dis_mat = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(i + 1, self.N):
                dis_mat[i, j] = np.square(data[i] - data[j]).sum()
                dis_mat[j, i] = dis_mat[i, j]
        return dis_mat

    def cal_weight_mat(self, dis_mat):
        # 计算相似性矩阵
        if self.criterion == 'gaussian': 
            if self.gamma is None:
                raise ValueError('gamma is not set')
            self.W = np.exp(-self.gamma * dis_mat)
        elif self.criterion == 'k_nearest':  
            if self.k==0 :
                self.W = np.exp(-self.gamma * dis_mat)
            else:
                if self.k is None or self.gamma is None:
                    raise ValueError('k or gamma is not set')
                self.W = np.zeros((self.N, self.N))
                for i in range(self.N):
                    inds = np.argpartition(dis_mat[i], self.k + 1)[:self.k + 1]  
                    tmp_w = np.exp(-self.gamma * dis_mat[i][inds])
                    self.W[i][inds] = tmp_w
                for i in range(self.N):
                    for j in range(i + 1, self.N):
                        if  self.W[i,j]==0 and self.W[j,i]!=0 :
                            self.W[i,j]=self.W[j,i]=0
        elif self.criterion == 'eps_nearest':  
            if self.dis_epsilon is None:
                raise ValueError('epsilon is not set')
            self.W = np.zeros((self.N, self.N))
            for i in range(self.N):
                inds = np.where(dis_mat[i] < self.dis_epsilon)
                self.W[i][inds] = 1.0 / len(inds)
        else:
            raise ValueError('the criterion is not supported')
        return

    def fit(self, data):
        self.init_param(data)
        if self.method == 'unnormalized':
            w, v = np.linalg.eig(self.L)
            inds = np.argsort(w)[:self.n_cluster]
            Vectors = v[:, inds]
        elif self.method == 'normalized':
            D = np.linalg.inv(np.sqrt(self.D))
            L = np.mat(D)* np.mat(self.L) * np.mat(D)
            w, v = np.linalg.eig(L)
            inds = np.argsort(w)[:self.n_cluster]
            Vectors = v[:, inds]
            normalizer = np.linalg.norm(Vectors, axis=1)
            normalizer = np.repeat(np.transpose([normalizer]), self.n_cluster, axis=1)
            Vectors = Vectors / normalizer
        else:
            raise ValueError('the method is not supported')
        km = KMeans(n_clusters=self.n_cluster, tol=self.tol,max_iter= self.max_iter,)
        km.fit(Vectors)
        self.labels=km.labels_
        return


if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from itertools import cycle
    from matplotlib import pyplot as plt

    data, label = make_blobs(centers=2, n_features=10, cluster_std=1.2, n_samples=500, random_state=1)
    sp = Spectrum(n_cluster=2, method='unnormalized', criterion='gaussian', gamma=0.1)
    sp.fit(data)
    

