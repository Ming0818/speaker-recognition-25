# -*- coding: utf-8 -*-
# traitement global des fichiers wav
import os,numpy as np,octaveIO as oio,string,subprocess
import fisher


def createDataFiles():
    if not os.path.exists('data'):
        os.makedirs('data')
        print "Please add some data, I don't work for free"
    else:
        res = []
        for root, dirs, files in os.walk('data'):
            print root,dirs,files
            for file in files:
                if file.endswith(".wav"):
                    print "treating file "+file
                    name=os.path.splitext(file)[0]
                    fileName = os.path.join(root, name)
                    wavName = fileName+'.wav'
                    matName = fileName+'.mat'
                    #print string.join(['octave','--eval','cepstraux('+'\''+wavName+'\',\''+matName+'\')'])
                    subprocess.call(['octave','--eval','cepstraux('+'\''+wavName+'\',\''+matName+'\')'])
                    triplet=oio.retrieve(matName,['c','mu','sig','pi'])
                    res.append(triplet)
        return res    

def gmms(data):
    c0 = np.concatenate([c for (c,mu,sig,pi) in data])
    mfccFile = 'data/grossmMat.mat'
    matName = 'data/grossMatOut.mat'
    oio.write(mfccFile,c0,'c')
    subprocess.call(['octave','--eval','gmm(\''+mfccFile+'\', \''+matName+'\')'])
    triplet0=oio.retrieve(matName,['mu','sig','pi'])

# res = createDataFiles()
# print res

# gmms(res)


def build_labels(name,dic,mu,pi):
    size = sum(len(val) for val in dic.itervalues())
    res = np.zeros(shape=(size,))
    for key in dic:
        if key==name:
            val=1
        else:
            val=-1
        for i in dick[key]:
            res[i]=val

def train(name,mu0,sig0,mu,pi):
    y = build_labels(name,dic,mu,pi)
    x=range(len(y))
    k = lambda i,j : fisher.K(x,i,j,mu[i],mu[j],sig0,pi[i],pi[j],mu0)
    w = ker_perceptron(x,y,k)
    return w
