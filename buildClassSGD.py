# -*- coding: utf-8 -*-
from __future__ import division #division en flottants par défaut
import numpy as np
import pickle
import preprocess
from buildLabels import buildLabels
from split import split
import sgd

def buildClassifier(subset, name, cs, iterations=100, dataDir='data', det=False, verbose=0):
    if (verbose >= 1):
        print 'Enter buildClassifier'
    if det:
        random.seed(1)

    evalC=[]
    bC=0.0
    evbC=0.0
    
    mfccs, mfccMatching, _, _, _ = preprocess.preprocess(subset, dataDir=dataDir, verbose=verbose)
    y = buildLabels(name, mfccMatching)
    dicL, dicV, dicT = split(mfccMatching, name, verbose=verbose)
    # learning data
    xL=[mfccs[k] for k in np.concatenate(dicL.values())]
    yL=[y[k] for k in np.concatenate(dicL.values())]
    # validation data
    xV=[mfccs[k] for k in np.concatenate(dicV.values())]
    yV=[y[k] for k in np.concatenate(dicV.values())]
    # testing data
    xT=[mfccs[k] for k in np.concatenate(dicT.values())]
    yT=[y[k] for k in np.concatenate(dicT.values())]

    for c in cs:
        if (verbose >= 1):
            print 'Processing C: ', c
            print 'Learning...'
        w = sgd.sgd(xL, yL, np.zeros(len(xL[0])+1), iterations, 1, sgd.L, 0.01, c)
        if (verbose >= 1):
            print 'Evaluating...'
        ev=sgd.eval(xV, yV, w[:-1], w[-1])
        evalC.append(ev)
        if (ev > evbC):
            evbC=ev
            bC=c

    if (verbose >= 1):
        print 'Building classifier with C:', bC, '...'
    xL2=xL+xV
    yL2=yL+yV
    w = sgd.sgd(xL2, yL2, np.zeros(len(xL[0])+1), iterations, 1, sgd.L, 0.01, bC)
    if (verbose >= 1):
        print 'Evaluating classifier...'
    ev=sgd.eval(xT, yT, w[:-1], w[-1])
    def f(wavFile):
        x = preprocess.mfcc(wavFile)
        tot=len(x)
        ok=0.0
        for i in xrange(tot):
            ok += np.dot(w[:-1], x[i]) + w[-1]
        return int(ok/tot > 0)
    if (verbose >= 1):
        print 'Exit buildClassifier'
    return f, ev, evalC
