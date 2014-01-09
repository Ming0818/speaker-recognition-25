# -*- coding: utf-8 -*-
from __future__ import division #division en flottants par défaut

# Preprocessing primitives

import os
import numpy as np
import octaveIO as oio
import subprocess as sp
from sklearn import mixture
import string

def gmm(x, nbG):
    g=mixture.GMM(n_components=nbG)
    g.fit(x)
    return g

def mfcc(wavFile, verbose=0):
    mfccFile='data/tmpmfcc.mat'
    command=['octave', '--silent', '--eval','cepstraux('+'\''+wavFile+'\',\''+mfccFile+'\')']
    if (verbose >= 2):
        print string.join(command)
    sp.call(command)
    c=oio.retrieve(mfccFile,['c'])[0]
    return c.tolist()

#Preprocess the available data (restricted to subset).
#Input: (subset, nbC=100, nbG=30, gmm=False)
#  subset: restrict the data set to the names in subset
#  if gmm is True:
#    nbC is the number of frames for each gmm computation
#    nbG is the number of gaussians in gmm computation
#Output: (mfccs, mfccMatching, mu, pi, gmmMatching)
#  mfccs: vector of mfccs
#  mfccMatching: dictionary matching names in subset to indices for mfccs
#  mu, pi and gmmMatching are empty if gmm is False
#  mu: means in gmm fitting
#  pi: normalized ditributions into the gaussians
#  gmmMatching: dictionary matching names in subset to indices in mu (and pi cause they are equal)
def preprocess(subset, nbC=100, nbG=30, ngmm=False, dataDir='data', verbose=0):
    if (verbose >= 1):
        print 'Enter preprocess'
    if not os.path.exists(dataDir):
        print 'Missing data'
        return [], {}, [], [], {}
    else:
        nbC2=nbC
        mfccs=[] #contiendra l'ensemble des mfccs
        mfccMatching={}
        mu=[]
        pi=[]
        gmmMatching={}
        numM=0
        numG=0
        for root, dirs, files in os.walk(dataDir):
            for file in files:
                if file.endswith('.wav'):
                    nameInDic=os.path.split(root)[-1]
                    if (verbose >= 1):
                        print string.join(['Processing', nameInDic+'/'+file])
                    if subset and not (nameInDic in subset):
                        if (verbose >= 1):
                            print 'Nothing to be done'
                        continue
                    name=os.path.splitext(file)[0]
                    fileName = os.path.join(root, name)
                    wavFile = fileName+'.wav'
                    # mfccFile = fileName+'mfcc.mat' #contient 'c'
                    if (verbose >= 1):
                        print 'Computing mfccs...'
                    c = mfcc(wavFile, verbose=verbose)
                    if (verbose >= 1):
                        print 'Done'
                    l=np.size(c, 0) #nombre total de frames
                    if not ngmm:
                        if not (mfccMatching.has_key(nameInDic)):
                            mfccMatching[nameInDic]=[]
                        mfccs = mfccs + c
                        mfccMatching[nameInDic]=mfccMatching[nameInDic]+range(numM, numM+l)
                        numM+=l
                    else:
                        if (nbC2==0):
                            nbv=1
                            nbC=l
                        else:
                            nbv=int(l/nbC)
                        if(not gmmMatching.has_key(nameInDic)):
                            gmmMatching[nameInDic]=[]
                        for j in xrange(nbv):
                            binf=j*nbC
                            bsup=(j+1)*nbC
                            newvf=c[binf:bsup] #vecteur de frames à ajouter
                            mfccs = mfccs + newvf #ajout dans l'ensemble global
                                
                        #Calcul des gmm
                            if (verbose >= 1):
                                print 'Computing GMM fitting...'
                                print str(nbC)+' points for '+str(nbG)+' gaussians'
                            if (verbose >= 2):
                                print newvf, nbG
                            g = gmm(newvf, nbG)
                            if (verbose >= 1):
                                print 'Done'
                            mu_j = g.means
                            pi_j = g.weights
                            
                            mu.append(mu_j)
                            pi.append(pi_j)
                            gmmMatching[nameInDic].append((numG, numM, numM+nbC))
                            numG+=1
                            numM+=nbC
    if (verbose >= 1):
        print 'Exit preprocess'
    return mfccs, mfccMatching, mu, pi, gmmMatching    
