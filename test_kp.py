# -*- coding: utf-8 -*-
from __future__ import division #division en flottants par défaut

import random
import numpy as np
import kernel_perceptron as kp

xApp=[random.uniform(-20, 20) for i in xrange(20)]
yApp=[random.uniform(-20, 20) for i in xrange(20)]
xyApp=np.transpose([xApp, yApp])
labelApp=[(-1)**(1 + (xApp[i]-yApp[i] >= 1)) for i in xrange(20)]
def k(x, y):
    return np.dot(x, y)

print xyApp
print labelApp
alp,b= kp.kp(xyApp, labelApp, k)

w=np.dot(alp, xyApp)
lPred=[np.dot(w, xyApp[i])+b for i in xrange(20)]
print alp
print w, b
print lPred

lPredb=[np.dot([1, -1], xyApp[i]) - 1 for i in xrange(20)]
print lPredb
