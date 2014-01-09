# -*- coding: utf-8 -*-
from __future__ import division #division en flottants par défaut
import numpy as np
import random, time
import preprocess as pp, buildLabels as bl, split, fisher, quad_prog as qp

# nC classes
# nS signals per class
# nF frames per signal
# N features per frames
# the means for i-th class are randomly chosen from [m[i], M[i]]
# nbG gaussians in gmm
# det=True to repeat the same generation
def buildData(nC=2, nS=100, nF=1000, N=10, m=[-50.0, 0.0], M=[0.0, 50.0], nbG=30, det=False):
    if det:
        random.seed(1)
    mfccs=[]
    mfccMatching={}
    mu=[]
    pi=[]
    gmmMatching={}
    numM=0
    numG=0
    for c in xrange(nC):
        mfccMatching[str(c)]=[]
        gmmMatching[str(c)]=[]
        muV=[[[] for g in xrange(nbG)] for i in xrange(nS)]
        piV=[np.zeros(nbG) for i in xrange(nS)]
        for i in xrange(nS):
            signal=[]
            pT=0.0
            for g in xrange(nbG):
                tmp=random.random()
                piV[i][g]=tmp
                pT+=tmp
                muV[i][g]=[m[c]+random.random()*(M[c]-m[c]) for k in xrange(N)]
            piV[i]=[pig/pT for pig in piV[i]]

            k=0.0
            g=0
            for p in piV[i]:
                ind=xrange(int(k*nF), int((k+p)*nF))
                tmp=[[random.gauss(muV[i][g][k], 1.0) for k in xrange(N)] for j in ind]
                signal+=tmp
                k+=len(ind)
                g+=1
            
            mfccs+=signal
            mfccMatching[str(c)]+=range(numM, numM+nF)
            gmms=pp.gmm(signal, nbG)
            mu.append(gmms.means)
            pi.append(gmms.weights)
            gmmMatching[str(c)].append((numG, numM, numM+nF))
            
            numM+=nF
            numG+=1

    return mfccs, mfccMatching, mu, pi, gmmMatching

def testQP(nbGs, nC=2, nS=100, nF=1000, N=10, m=[-50.0, 0.0], M=[0.0, 50.0], det=False, verbose=0):
    if det:
        t=1
    else:
        t=time.clock()

    evalG=[]
    bG=0
    evbG=0.0
    name='1'
    
    for nbG in nbGs:
        if (verbose >= 1):
            print 'Processing with '+str(nbG)+' gaussians...'
        random.seed(t)
        mfccs, _, mu, pi, gmmMatching = buildData(nC, nS, nF, N, m, M, nbG)
        if (verbose >= 2):
            print gmmMatching
        y = bl.buildLabels2(name, gmmMatching)
        dicL, dicV, dicT = split.split(gmmMatching, name, verbose=verbose)
        # learning data
        dicLv=np.concatenate(dicL.values())
        xL=[j for (j,_,_) in dicLv]
        yL=[y[j] for (j,_,_) in dicLv]
        # validation data
        dicVv=np.concatenate(dicV.values())
        xV=[j for (j,_,_) in dicVv]
        yV=[y[j] for (j,_,_) in dicVv]
        # testing data
        dicTv=np.concatenate(dicT.values())
        xT=[j for (k,_,_) in dicTv]
        yT=[y[j] for (j,_,_) in dicTv]
        if (verbose >= 1):
            print 'xL:', np.shape(xL)
            print 'dicLv:', np.shape(dicLv)
            print 'mfccs:', np.shape(mfccs)
            print np.concatenate([range(u,v) for (_,u,v) in dicLv])
        g = pp.gmm([mfccs[j] for j in np.concatenate([range(u,v) for (_,u,v) in dicLv])], nbG)
        mu0=g.means
        sig0=g.covars

        def k(i,j):
            res = fisher.K(xL,i,j,mu[i],mu[j],sig0,pi[i],pi[j],mu0)
            return res
            
        if (verbose >= 1):
            print 'Learning...'
            print 'Evaluating...'
        w, b, ev = qp.qp(xL, yL, xV, yV, k, 1.0)
        ev=ev[0][0]
        evalG.append(ev)
        if (ev > evbG):
            evbG=ev
            bG=nbG

    if (verbose >= 1):
        print 'Building classifier with '+str(bG)+' gaussians...'
    xL2=xL+xV
    yL2=yL+yV
    dicL2v=np.concatenate([dicLv,dicVv])
    g2 = pp.gmm([mfccs[k] for k in np.concatenate([range(u,v) for (_,u,v) in dicL2v])], bG)
    mu02=g2.means
    sig02=g2.covars
    if (verbose >= 1):
        print len(dicL2v), np.shape(mu), np.shape(sig02)
    def k2(i,j):
        res = fisher.K(dicL2v,i,j,mu[i],mu[j],sig02,pi[i],pi[j],mu02)
        return res
         
    if (verbose >= 1):
        print 'Learning...'
        print 'Evaluating...'
    w2, b2, ev2 = qp.qp(xL2, yL2, xT, yT, k2, 1.0)
    ev2=ev2[0][0]
    if (verbose >= 2):
        print mu, pi, mu02, sig02
    print bG, evbG,ev2, evalG
