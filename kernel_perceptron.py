# -*- coding: utf-8 -*-
from __future__ import division #division en flottants par défaut
import sys
import numpy as np
import random
import time
import plot

def ker_perceptron(x, y, k):
    stop = 10000000
    n = len(x)
    w = np.zeros(shape=(n,))
    w[0] = y[0]
    b = y[0]
    valeurs = np.array([y[i] * (w[0]*k(x[0], x[i]) + b) for i in xrange(n)])
    negatifs = [i for i in xrange(n) if valeurs[i] < 0]
    while(negatifs and (stop > 0)):
        j = negatifs[0]
        w[j] = w[j] + y[j]
        b = b + y[j]
        valeurs = np.add(valeurs, np.array([y[i]* y[j]*(k(x[j], x[i]) + 1) for i in xrange(n)]))
        negatifs = [i for i in xrange(n) if valeurs[i] < 0]
        stop -= 1
        plot.plot(x, y, w, b)
    return w, b


