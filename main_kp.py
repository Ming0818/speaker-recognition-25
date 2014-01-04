# -*- coding: utf-8 -*-
# traitement global des fichiers wav
import os,numpy as np,octaveIO as oio,string,subprocess
import fisher
import random
import kernel_perceptron as kp
from sklearn import mixture

def gmm(x, nbG):
    g=mixture.GMM(n_components=nbG)
    g.fit(x)
    return g

def createDataFiles(nbc, nbG):
    if not os.path.exists('data'):
        os.makedirs('data')
        print "Please add some data, I don't work for free"
    else:
        mfccs = [] #contiendra l'ensemble des mfccs
        muApp=[]
        piApp=[]
        dicApp={}
        numApp=0
        muTest=[]
        piTest=[]
        dicTest={}
        numTest=0
        for root, dirs, files in os.walk('data'):
            #print root,dirs,files
            for file in files:
                if file.endswith(".wav"):
                    print "treating file "+file
                    nameInDic=os.path.split(root)[-1]
                    name=os.path.splitext(file)[0]
                    fileName = os.path.join(root, name)
                    wavFile = fileName+'.wav'
                    mfccFile = fileName+'mfcc.mat' #contient 'c'
                    #print string.join(['octave','--eval','cepstraux('+'\''+wavFile+'\',\''+mfccFile+'\')'])
                    subprocess.call(['octave', '--silent', '--eval','cepstraux('+'\''+wavFile+'\',\''+mfccFile+'\')'])
                    c=oio.retrieve(mfccFile,['c'])[0]
                    l=np.size(c, 0) #nombre total de frames
                    nbv=l/nbc
                    if(not dicApp.has_key(nameInDic)):
                        dicApp[nameInDic]=[]
                        dicTest[nameInDic]=[]
                    for j in xrange(min(nbv, 5)):
                        binf=j*nbc
                        bsup=(j+1)*nbc
                        newvf=c[binf:bsup] #vecteur de frames à ajouter
                        mfccs.append(newvf) #ajout dans l'ensemble global

                        #Calcul des gmm
                        g = gmm(newvf, nbG)
                        mu_j = g.means
                        pi_j = g.weights

                        muApp.append(mu_j)
                        piApp.append(pi_j)
                        dicApp[nameInDic].append(numApp)
                        numApp+=1
                    for j in xrange(min(nbv, 5), nbv):
                        binf=j*nbc
                        bsup=(j+1)*nbc
                        newvf=c[binf:bsup] #vecteur de frames à ajouter

                        #Calcul des gmm
                        g = gmm(newvf, nbG)
                        mu_j = g.means
                        pi_j = g.weights

                        muTest.append(mu_j)
                        piTest.append(pi_j)
                        dicTest[nameInDic].append(numTest)
                        numTest+=1
        mu=muApp+muTest
        pi=piApp+piTest
    return mfccs, mu, pi, dicApp, dicTest, numApp    

def gmms(data, nbG):
    c0 = np.concatenate(data)
    g=gmm(c0, nbG)
    mu0=g.means
    sig0=g.covars
    return mu0, sig0

def build_labels(name,dic):
    size = sum(len(val) for val in dic.itervalues())
    res = np.zeros(size)
    for key in dic:
        if key==name:
            val=1
        else:
            val=-1
        for i in dic[key]:
            res[i]=val
    return res

def make_training_set(name,dic,m):
    """Take m random adversaries to help train <name>"""
    nameKeys=dic[name]
    advKeys=[dic[key] for key in dic if key<>name]
    advs = random.sample(advKeys,min(len(advKeys),m))
    return {name : nameKeys, '#mechant' : advs}

def train(name,mu0,sig0,mu,pi,dic):
    print('learning ' + name)
    y = build_labels(name,dic)
    x=range(len(y))
    def k(i,j):
        res = fisher.K(x,i,j,mu[i],mu[j],sig0,pi[i],pi[j],mu0)
        return res
    w, b = kp.kp(x,y,k)
    # w = qp(y, 1, k)
    return w, b

nbG=4
mfccs, mu, pi, dicApp, dicTest, numApp = createDataFiles(100, nbG)
#print res
print 'GMM sur l\'ensemble des points\n'

mu0, sig0 = gmms(mfccs, nbG)

w, b = train('sarkozy', mu0, sig0, mu, pi, dicApp)

def predKP(w, b, mu, pi, mu0, sig0, dic, num):
    print 'prediction...'
    yPred=[]
    T=len(w)
    for name in dic:
        for i in dic[name]:
            #calcule <w,i-eme>
            v=[w[j]*fisher.K(w, j, i+num, mu[j], mu[i], sig0, pi[j], pi[j], mu0) for j in xrange(T)]
            tmp=sum(v) + b
            yPred.appen(tmp)
            #print name, i+num, ' : ', tmp
    return yPred
            
def evalKP(yPred, nameL, dic, seuil=0):
    ok = 0;
    tot = sum(len(val) for val in dic.itervalues())
    for name in dic:
        for i in dic[name]:
            if(name==nameL):
                ok+=(tmp-seuil > 0)
            else:
                ok+=(tmp-seuil < 0)
    return ok, ok/float(tot)

print 'evaluation...'
rho = []
yPred=predKP(w, b, mu, pi, mu0, sig0, dicTest, numApp)
for i in xrange(3):
    seuil = 2**(2*(-3+i))
    k, tmp = evalKP(yPred, 'sarkozy', dicTest, seuil)
    rho.append(tmp)
for i in xrange(4):
    seuil = 2**(3*i)
    k, tmp = evalKP(yPred, 'sarkozy', dicTest, seuil)
    rho.append(tmp)

print rho
