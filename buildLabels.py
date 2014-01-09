# -*- coding: utf-8 -*-
from __future__ import division #division en flottants par défaut
import numpy as np

def buildLabels(name,dic):
    size = sum(len(val) for val in dic.itervalues())
    res = np.zeros(size)
    for key in dic:
        if key==name:
            val=1
        else:
            val=-1
        for i in dic[key]:
            res[i]=val
    return res.tolist()


def buildLabels2(name,dic):
    size = sum(len(val) for val in dic.itervalues())
    res = np.zeros(size)
    for key in dic:
        if key==name:
            val=1
        else:
            val=-1
        for (i,_,_) in dic[key]:
            res[i]=val
    return res.tolist()
