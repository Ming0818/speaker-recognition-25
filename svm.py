# -*- coding: utf-8 -*-
from __future__ import division #division en flottants par défaut
import sys
import numpy as np
import random
import os, pickle
import plot
import kernel_perceptron as kp

# http://stackoverflow.com/questions/17784587/gradient-descent-using-python-and-numpy-machine-learning

def testDef(x):
    return False

class LossFun:
     def __init__(self, lf, grad):
         self.lossFun = lf
         self.grad = grad

def hinge(x):
    return max(0,1-x)

def HingeLoss(xi,yi,w): # b est la dernière coordonnée de w
    return hinge(yi * (np.dot(w[:-1],xi) + w[-1]))
    
        
def HLgrad(xi,yi,w,eps):
    evalfxi = yi * (np.dot(w[:-1],xi) + w[-1])
    delta = evalfxi - 1
    if delta > eps:
        res = np.zeros(shape=len(w))
    elif delta < -eps:
        res = (-yi)*(np.concatenate([xi,np.array([1])]))
    else:    
        res = (-yi/2.)*(np.concatenate([xi,np.array([1])]))
    # print "res de HLGrad : ", res, "sur ", xi, yi, w
    return(res)

L = LossFun(HingeLoss,HLgrad)

def sgd(x,y,w,Tmax,eta,L,eps,test=testDef):
    eta1 = eta
    t = 1
    theta = 0
    while(t <= Tmax and not(test(x))):   
        lossGrads = np.array([L.grad(x[i],y[i],w,eps) for i in xrange(len(x))])
        v = np.add (np.concatenate([w[:-1],np.array([0])]),lossGrads.sum(axis=0)/(len(x)))
        eta = eta1 / np.sqrt(t)
        w = np.subtract(w,eta * v)
        t = t+1
        theta = np.add(theta,w)
    res = theta / (t)
    return(res)

def genData(averages,N):
    x = np.zeros(shape=(2,N,2))
    y = np.concatenate ([np.ones(shape=N),  ((-1) * np.ones(shape=N))])
    for c in [0,1]:
        for i in xrange(N):
            for j in [0,1]:
                x[c][i][j] = averages[c] + random.uniform(-1,1)
    res = (np.concatenate([x[0],x[1]]),y)
    return res

file = "data"
curdir = os.path.dirname(os.path.realpath(__file__))+"/"

try:
    sampleSize = int(sys.argv[1])
except IndexError:
    sampleSize = 50

averages = [1,5]

if os.path.isfile(curdir+file):
    f = open(curdir+file,'r+')
    (x,y,size) = pickle.load(f)
    if size<>sampleSize:
        print 'Taille de l\'echantillon : ', sampleSize
        f.truncate()
        (x,y) = genData(averages,sampleSize)
        pickle.dump((x,y,sampleSize),f)
    f.close()
else:
    f = open(curdir+file,'w+')
    (x,y) = genData(averages,sampleSize)
    pickle.dump((x,y,sampleSize),f)
    f.close()

iterations = 400
w = sgd(x,y,np.zeros(len(x[0])+1),iterations,1,L,0.01)
# print "w",w

def classify(xi,w):
    return np.sign(np.dot(w[:-1],xi) + w[-1])
def printClassify(xi,w):
    return (np.dot(w[:-1],xi) + w[-1])


def testClassification(sampleSize,x,y,w):
    goodC = 0
    badC = 0
    for i in xrange(2*sampleSize):
#        print classify(x[i],w),y[i]
#        print "valeur: ", printClassify(x[i],w)
        if classify(x[i],w)==y[i]:
            goodC +=1
        else:
            badC += 1
#    print "Good : ", goodC
#    print "Bad : ", badC
    print "rate : ", goodC/sampleSize/2
    plot.plot(x, y, w[:-1], w[-1])

# w[2] = 0.7
# testClassification(sampleSize,x,y,w)
# print x
# print y
# print w

k = np.dot
w2 = kp.ker_perceptron(x, y, k)
w3 = 0
for i in xrange(len(x)):
    w3 = np.add(w3, w2[i] * x[i])
b = np.sum([w2[i] for i in xrange(len(w2))])
plot.plot(x,y,w3,b)
