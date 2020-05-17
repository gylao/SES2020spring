# -*- coding: utf-8 -*-
"""
Created on Thu May  7 19:49:30 2020

@author: gylao
"""

import numpy as np
import pandas as pd
import random

def logistic_func(beta, x):
    z = np.dot(x,beta)
    y = 1.0 / (1.0 + np.exp(-z))
    return y.reshape(-1,1)

def loss_func(beta, x, y):
    loss = 0
    size = x.shape[0]
    y_pred = logistic_func(beta, x)
    for i in range(size):
        loss = loss - np.log( y[i,0]*y_pred[i,0] + (1-y[i,0])*(1-y_pred[i,0]) )
    loss = loss / size
    return loss

def cal_lr(beta, x, y_pred):
    size = x.shape[0]
    Hess = 0
    for i in range(size):
        Hess = Hess + np.dot(x[i:i+1,:].T,x[i:i+1,:])*y_pred[i,0]*(1-y_pred[i,0])
    lr  = np.linalg.inv(Hess)
    return lr

def fit(x, y, Niter):
    w = np.random.normal(loc=0.0, scale=1.0, size=x.shape[1]-1)
    b = np.random.normal(loc=0.0, scale=1.0)
    #w = np.ones([x.shape[1]-1])
    #b = np.ones([1])
    beta = np.append(w,b).reshape(-1,1)
    size = x.shape[0]
    
    for i in range (Niter):
        y_pred = logistic_func(beta, x)
        
        #随机梯度下降 SGD
        index = random.randint(0,size-1)
        dbeta = (x[index]*(y_pred[index] - y[index])).reshape(-1,1)
        lr = 0.1
        beta = beta - lr*dbeta
        
#        #牛顿法        
#        dbeta = np.dot(x.T,y_pred - y)
#        dbeta = dbeta / size     
#        lr = cal_lr(beta, x, y_pred)
#        beta = beta - np.dot(lr,dbeta)
        
#        #固定步长梯度下降 BGD
#        dbeta = np.dot(x.T,y_pred - y)
#        dbeta = dbeta / size     
#        lr = 0.1
#        beta = beta - lr*dbeta
        
        if i % 1000 == 0:
            print('Loss is: ', loss_func(beta, x, y))
    
    print('Accuracy is: ', cal_acc(beta, x, y)[0])
    
    return beta
        
def cal_acc(beta, x, y):
    size = x.shape[0]
    y_pred = logistic_func(beta, x)
    y_pred = np.array([0 if y_pred[i] < 0.5 else 1 for i in range(size)])
    acc = np.mean([1 if y[i] == y_pred[i] else 0 for i in range(size)])
    return acc, y_pred
    
def undersampling(data):
    data0 = data[data[:,0]==0]
    data1 = data[data[:,0]==1]
    size0 = data0.shape[0]
    size1 = data1.shape[0]
    if size0 >= size1:
        index= np.random.choice(size0,size1,replace=False,p=None)
        data0 = data0[index,:]
    else:
        index= np.random.choice(size1,size0,replace=False,p=None)
        data1 = data1[index,:]
    x = np.concatenate((data0[:, 1:31], data1[:, 1:31]), axis=0)
    y = np.concatenate((data0[:, 0:1], data1[:, 0:1]), axis=0)
    return x, y

def oversampling(data):
    data0 = data[data[:,0]==0]
    data1 = data[data[:,0]==1]
    size0 = data0.shape[0]
    size1 = data1.shape[0]
    if size0 >= size1:
        index= np.random.choice(size1,size0-size1,replace=False,p=None)
        data1 = np.concatenate((data1, data1[index,:]), axis=0)
    else:
        index= np.random.choice(size0,size1-size0,replace=False,p=None)
        data0 = np.concatenate((data0, data0[index,:]), axis=0)
    x = np.concatenate((data0[:, 1:31], data1[:, 1:31]), axis=0)
    y = np.concatenate((data0[:, 0:1], data1[:, 0:1]), axis=0)
    return x, y

def pre_process(dataset):
    data = dataset.iloc[:, 1:32].values
    data[data=='M'] = 1
    data[data=='B'] = 0
    data = data.astype(np.float64)

    #x, y = undersampling(data)
    x, y = oversampling(data)
    
    size = x.shape[0]
    index = np.random.permutation(size)
    x = x[index]
    y = y[index]
    #标准化
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    x =  (x - mu) / sigma
    #凑成^x的形式 方便后续计算
    x = np.concatenate((x, np.ones([size,1])), axis=1)
    #划分训练集和测试集
    split = int(len(y)*0.7)
    x_train = x[:split]
    y_train = y[:split]
    x_test = x[split:]
    y_test = y[split:]  
    return x_train, y_train, x_test, y_test

def test(beta, x, y):
    acc, y_pred = cal_acc(beta, x, y)
    print('Test_Accuracy is: ', acc)
    return y_pred


dataset = pd.read_csv('./data/data.csv')

x_train, y_train, x_test, y_test = pre_process(dataset)

beta = fit(x_train, y_train, 10000)

y_pred_test = test(beta, x_test, y_test)


