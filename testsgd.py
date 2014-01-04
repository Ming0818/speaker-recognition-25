# -*- coding: utf-8 -*-
from __future__ import division #division en flottants par défaut

import random
import numpy as np
# import kernel_perceptron as kp
import sgd
import plot

def genData(averages,N):
    x = np.zeros(shape=(2,N,2))
    y = np.concatenate ([np.ones(shape=N),  ((-1) * np.ones(shape=N))])
    for c in [0,1]:
        for i in xrange(N):
            for j in [0,1]:
                x[c][i][j] = averages[c] + random.uniform(-1,1)
    res = (np.concatenate([x[0],x[1]]),y)
    return res

iterations=800
eps = 0.01
eta = 1
C=1
averages = [1,3]
sample=100
(xApp, yApp) = genData(averages, sample)
k=np.dot
w1 = sgd.sgd(xApp, yApp, np.zeros(len(xApp[0])+1),iterations,eta,sgd.L,eps,C)
w,b = w1[:-1],w1[-1]


# w=np.dot(alp, xApp)
yPred=[np.dot(w, xApp[i])+b for i in xrange(2*sample)]
# print np.multiply(yApp, yPred)

def makeTitle(iterations,eta,eps):
    res = 'SGD with '
    res = res + str(iterations) + ' iterations, '
    res = res + 'eta=' + str(eta)+ ', epsilon=' + str(eps)
    res = res + '\n'
    return res
    

plot.plot(xApp,yApp,w,b,makeTitle(iterations,eta,eps),'SGD')
