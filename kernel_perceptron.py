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


def kp(x, y, k, stop=100000, log=False, logFile=None, step=50):
    n=len(x)
    w=np.zeros(n)
    b=y[0]
    w[0] = y[0] #initialisation de w
    ind=0
    if ((not logFile) or (step <= 0)):
        log=False

    valeurs = [y[i]*(w[0]*k(x[i], x[0]) + b) for i in xrange(n)]
    negatifs = [i for i in xrange(n) if valeurs[i] < 0]
    while(negatifs and (stop > 0)):
        if (log and not (ind % step)):
            sFile=logFile+str(int(ind/step))+'.png'
            sTitle='Kernel perceptron at '+str(ind)+'-th iteration'
            plot.plot(x, y, np.dot(w, x), b, title=sTitle, saveFile=sFile, display=False)

        j=negatifs[0]
        w[j] = w[j] + y[j]
        b = b + y[j]
        deltaValeurs = [y[i]*(y[j]*k(x[i], x[j]) + y[j]) for i in xrange(n)]
        valeurs=np.add(valeurs, deltaValeurs)
        negatifs = [i for i in xrange(n) if valeurs[i] < 0]
        stop -= 1
        ind+=1
    if (stop==0):
        print 'Kernel perceptron didn\'t converge !'
    return w, b, ind
        
