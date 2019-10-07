# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:38:43 2019

@author: lenovo
"""

import csv
import numpy as np
def sigmoid(z):
    return 1/(1+np.exp(-z))
def init_data():
    data=np.loadtxt('HTRU_2_train.csv',delimiter=',')
    line,width=np.shape(data)
    target=data[:,-1]
    data_1=data[:,0:-1]
    dataIn=np.insert(data_1,0,1,axis=1)
    return dataIn,target
def grad_descent(dataIn,target):
    dataMatrix=np.mat(dataIn)
    labelMat=np.mat(target).transpose()
    line1,width1=np.shape(dataMatrix)
    weights=np.ones((width1,1))
    alpha=0.001               #0.006  0.0003  43   48    51  52   65   69  7  8   85  94  96   99
    maxCycle=10000       #1000  2000
    for i in range(maxCycle):
        h=sigmoid(dataMatrix*weights)
        weights=weights-alpha*dataMatrix.transpose()*(h-labelMat)
    return weights,h
dataIn,target=init_data()
weights,h=grad_descent(dataIn,target)
test=np.loadtxt('HTRU_2_test.csv',delimiter=',')
number,width2=np.shape(test)
text_1=[0]*number
for i in range(number):
    if(sigmoid(np.insert(test[i],0,1)*weights)>0.5):
        text_1[i]=1
    else:
        text_1[i]=0

#print(number)
rows_first=['id','y']
rows=np.array([[0,0]]*number)

for i in range(number):
    rows[i][0]=i+1
    rows[i][1]=text_1[i]
#print(rows)
with open('test.csv','w',newline='') as csv_file:
    writer=csv.writer(csv_file)
    writer.writerow(rows_first)
    for row in rows:
        writer.writerow(row)
csv_file.close()
#with open('test.csv','r') as read_file:
#    reader=csv.reader(read_file)
#    print([row for row in reader])
#https://github.com/mawenyi1011/mwyWEB/blob/master/%E7%AB%9E%E8%B5%9B%E4%B8%80.docx
#data=np.loadtxt('HTRU_2_train.csv',delimiter=',')
#a=0
#shape1,shape2=data.shape
#for i in range(652):
#    if(sigmoid(np.insert(data[i,0:-1],0,1)*weights)>0.5):
#        if(data[i,-1]==1):
#            a=a+1
#    else:
#        if(data[i,-1]==0):
#            a=a+1
#print(a/(652))















