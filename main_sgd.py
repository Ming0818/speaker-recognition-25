# -*- coding: utf-8 -*-
# traitement global des fichiers wav
import os,numpy as np,octaveIO as oio,string,subprocess
import sgd
import random

def createDataFiles(nbc):
    if not os.path.exists('data'):
        os.makedirs('data')
        print "Please add some data, I don't work for free"
    else:
        xApp = [] #contiendra l'ensemble des mfccs à apprendre
        dicApp={}
        numApp=0
        xTest=[]
        dicTest={}
        numTest=0
        for root, dirs, files in os.walk('data'):
            for file in files:
                if file.endswith(".wav"):
                    print "treating file "+file
                    nameInDic=os.path.split(root)[-1]
                    name=os.path.splitext(file)[0]
                    fileName = os.path.join(root, name)
                    wavFile = fileName+'.wav'
                    mfccFile = fileName+'mfcc.mat' #contient 'c'
                    subprocess.call(['octave', '--silent', '--eval','cepstraux('+'\''+wavFile+'\',\''+mfccFile+'\')'])
                    c=oio.retrieve(mfccFile,['c'])[0]
                    l=np.size(c, 0) #nombre total de frames
                    nbv=l/nbc
                    if(not dicApp.has_key(nameInDic)):
                        dicApp[nameInDic]=[]
                        dicTest[nameInDic]=[]
                    for j in xrange(min(nbv, 2)):
                        binf=j*nbc
                        bsup=(j+1)*nbc
                        newvf=c[binf:bsup] #vecteur de frames à ajouter
                        xApp = xApp + newvf.tolist() #ajout dans l'ensemble global
                        dicApp[nameInDic] = dicApp[nameInDic] + range(numApp, numApp + nbc)
                        numApp += nbc
                    for j in xrange(min(nbv, 2), min(nbv, 4)):
                        binf=(j-2)*nbc
                        bsup=(j-1)*nbc
                        newvf=c[binf:bsup] #vecteur de frames à ajouter
                        xTest = xTest + newvf.tolist() #ajout dans l'ensemble global
                        dicTest[nameInDic] = dicTest[nameInDic] + range(numTest, numTest + nbc)
                        numTest += nbc
        return xApp, dicApp, xTest, dicTest

def build_labels(name,dic):
    size = sum(len(val) for val in dic.itervalues())
    res = np.zeros(shape=(size,))
    for key in dic:
        if key==name:
            val=1
        else:
            val=-1
        for i in dic[key]:
            res[i]=val
    return res

# def make_training_set(name,dic,m):
#     """Take m random adversaries to help train <name>"""
#     nameKeys=dic[name]
#     advKeys=[dic[key] for key in dic if key<>name]
#     advs = random.sample(advKeys,min(len(advKeys),m))
#     return {name : nameKeys, '#mechant' : advs}

def trainSgd(name, dic, x):
    y = build_labels(name,dic)
    iterations = 400
    w = sgd.sgd(x,y,np.zeros(len(x[0])+1),iterations,1,sgd.L,0.01)
    return w

def evalSgd(nameL, w, b, dic, x):
    tot = len(x)
    ok = 0
    for name in dic:
        for i in dic[name]:
            tmp = np.dot(w, x[i]) + b
            if (name == nameL):
                ok += (tmp > 0)
            else:
                ok += (tmp < 0)
            #print name, i, ' : ', tmp
    return ok, ok/float(tot)

print 'create data files...'
xApp, dicApp, xTest, dicTest = createDataFiles(100)
print 'launch sgd...'
w2 = trainSgd('sarkozy', dicApp, xApp)
print 'evaluation...'
k, rho = evalSgd('sarkozy', w2[:-1], w2[-1], dicTest, xTest)

print k, rho
