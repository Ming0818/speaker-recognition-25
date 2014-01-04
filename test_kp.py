# -*- coding: utf-8 -*-
from __future__ import division #division en flottants par défaut

import random
import numpy as np
import kernel_perceptron as kp
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

averages = [1,5]
sample=20
(xApp, yApp) = genData(averages, sample)
k=np.dot
alp,b= kp.kp(xApp, yApp, k)

w=np.dot(alp, xApp)
yPred=[np.dot(w, xApp[i])+b for i in xrange(2*sample)]
# print np.multiply(yApp, yPred)

plot.plot(xApp,yApp,w,b)
