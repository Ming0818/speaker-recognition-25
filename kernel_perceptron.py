# -*- coding: utf-8 -*-
from __future__ import division #division en flottants par défaut
import sys
import numpy as np
import random
import time
import plot

def ker_perceptron(oldX, y, k):
    stop = 1000000
    n = len(oldX)
    x = np.array([np.append(oldX[i],1) for i in xrange(n)])
    w = np.zeros(shape=(n,))
    w[0] = y[0]
    valeurs = np.array([y[i] * w[0] * k(x[0], x[i]) for i in xrange(n)])
    #print valeurs
    negatifs = [i for i in xrange(n) if valeurs[i] < 0]
    while(negatifs and (stop > 0)):
        j = negatifs[0]
        w[j] = w[j] + (y[j])
        deltaValeurs = np.array([y[i]* (y[j])*(k(x[j], x[i])) for i in xrange(n)])
        valeurs = np.add(valeurs, deltaValeurs)
        #print valeurs
        negatifs = [i for i in xrange(n) if valeurs[i] < 0]
        stop -= 1
    return w


