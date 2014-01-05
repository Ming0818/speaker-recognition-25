# -*- coding: utf-8 -*-
from __future__ import division #division en flottants par défaut
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

def plot(x, y, w, b,title=None,saveFile=None, display=True):
    x1max = np.max(x[:, 0]) + 2
    x2max = np.max(x[:, 1]) + 2
    x1min = np.min(x[:, 0]) - 2
    x2min = np.min(x[:, 1]) - 2

    s1 = [x[i] for i in xrange(len(x)) if y[i] >= 0]
    s2 = [x[i] for i in xrange(len(x)) if y[i] < 0]
    s11 = [s1[i][0] for i in xrange(len(s1))]
    s12 = [s1[i][1] for i in xrange(len(s1))]
    s21 = [s2[i][0] for i in xrange(len(s2))]
    s22 = [s2[i][1] for i in xrange(len(s2))]

    if (w[1] <> 0):
        l1 = [x1min, x1max]
        l2 = [-(b + w[0]*x1min)/w[1], -(b + w[0]*x1max)/w[1]]
    else:
        l1 = [-b/w[0], -b/w[0]]
        l2 = [x2min, x2max]

    plt.plot(s11, s12, 'ro', s21, s22, 'bx', l1, l2, 'g-')
    if title:
        plt.title(title)
    plt.axis([x1min, x1max, x2min, x2max])
    if saveFile:
        plt.savefig(saveFile)
    if display:
        plt.show()

def plotFindC(res,legend,e):
    plt.plot(np.log2(res[0,:]),res[1,:], 'r--')
    # plt.show()
    plt.ylabel('accuracy')
    plt.xlabel('log C')
    # plt.text(4, e - 0.1, legend)
    plt.title(legend)
    plt.subplots_adjust(top=0.7)
    plt.savefig('bestC.png')

def plot_res(fileName, title):
    random.seed(100)
    res=pickle.load(open(fileName, 'rb'))
    acc={}
    nbGs={}
    for (name, nbG, C) in res:
        if (not name in acc):
            acc[name]=[]
            nbGs[name]=[]
        acc[name].append(res[(name, nbG, C)][0][0])
        nbGs[name].append(nbG)
    for name in acc:
        l=len(nbGs[name])
        tmp=sorted(range(l), key=lambda k:nbGs[name][k])
        plt.plot([nbGs[name][i] for i in tmp], [acc[name][i] for i in tmp], color=(random.random(), random.random(), random.random()))
    plt.axis([0,110,0.4,1.1])
    plt.ylabel('accuracy')
    plt.xlabel('number of gaussians')
    plt.title(title)
    plt.subplots_adjust(top=0.9)
    plt.savefig('opt_qp.png')
    plt.show()

# plot_res('resultat.dat', 'Optimisation du nombre de gaussiennes des GMM\n pour l\'optimisation quadratique')
        
