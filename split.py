# -*- coding: utf-8 -*-
from __future__ import division #division en flottants par défaut
import random

#Split the dictionary into 3 disjoint dictionaries
#A dictionary is balanced if the number of name's objects is roughly equal to the number of non-name's objects
#Input: (dic, name)
#  dic: the dictionary to be split
#  name: giving the ratio of splitting 
#    -> 4/9 of name's objects
#    -> 2/9 of name's objects 
#    -> 1/3 of name's objects
#Output: (dic1, dic2, dic3)
#  dic1: contains 4/9 of name's objects and balanced
#  dic2: contains 2/9 of name's objects and balanced
#  dic3: contains the remaining objects

def split(dic, name, verbose=0, det=False):
    if (verbose >= 1):
        print 'Enter split'
    if det:
        random.seed(1)
    nameObj=[k for k in dic[name]]
    nameNonObj=[]
    for k in dic:
        if k <> name:
            nameNonObj=nameNonObj + [(k, i) for i in dic[k]]
    l=len(nameObj)
    n1= int(4*l/9)
    n2= int(2*l/9)
    if (n2==0):
        print 'Not enough data for '+name+' entry'
        return {}, {}, {}
    if (len(nameNonObj) < n1+n2):
        print 'Not enough data for non-'+name+' entries'
        return {}, {}, {}
    dic1={}
    dic2={}
    dic3={}
    # build dic1
    l1=random.sample(nameObj, n1)
    l2=random.sample(nameNonObj, n1)
    if (verbose >= 2):
        print 'dic1: ', l1, l2
    dic1[name] = l1
    for i in l1:
        nameObj.remove(i)
    for (k, i) in l2:
        if not (dic1.has_key(k)):
            dic1[k]=[]
        dic1[k].append(i)
        nameNonObj.remove((k,i))
    if (verbose >= 2):
        print nameObj, nameNonObj

    # build dic2
    l1=random.sample(nameObj, n2)
    l2=random.sample(nameNonObj, n2)
    if (verbose >= 2):
        print 'dic2: ', l1, l2
    dic2[name] = l1
    for i in l1:
        nameObj.remove(i)
    for (k, i) in l2:
        if not (dic2.has_key(k)):
            dic2[k]=[]
        dic2[k].append(i)
        nameNonObj.remove((k,i))
    if (verbose >= 2):
        print nameObj, nameNonObj

    # build dic3
    if (verbose >= 2):
        print 'dic3: ', nameObj, nameNonObj
    dic3[name]=nameObj
    for (k,i) in nameNonObj:
        if not (dic3.has_key(k)):
            dic3[k]=[]
        dic3[k].append(i)
    if (verbose >= 2):
        print nameObj, nameNonObj

    if (verbose >= 1):
        print 'Exit split'
    return dic1, dic2, dic3
