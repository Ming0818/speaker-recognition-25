# -*- coding: utf-8 -*-
# traitement global des fichiers wav
import os,numpy as np,octaveIO as oio,string,subprocess
import fisher
import random
from sklearn import mixture
from quad_prog import qp

def gmm(x, nbG):
    g=mixture.GMM(n_components=nbG)
    g.fit(x)
    return g

def createDataFiles(nbc, nbG):
    if not os.path.exists('data'):
        os.makedirs('data')
        print "Please add some data, I don't work for free"
    else:
        mfccs=[] #contiendra l'ensemble des mfccs
        dic={}
        mu=[]
        pi=[]
        num=0
        for root, dirs, files in os.walk('data'):
            #print root,dirs,files
            for file in files:
                if file.endswith(".wav"):
                    #print "treating file "+file
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
                    if(not dic.has_key(nameInDic)):
                        dic[nameInDic]=[]
                    for j in xrange(nbv):
                        binf=j*nbc
                        bsup=(j+1)*nbc
                        newvf=c[binf:bsup] #vecteur de frames à ajouter
                        mfccs.append(newvf) #ajout dans l'ensemble global

                        #Calcul des gmm
                        g = gmm(newvf, nbG)
                        mu_j = g.means
                        pi_j = g.weights

                        mu.append(mu_j)
                        pi.append(pi_j)
                        dic[nameInDic].append(num)
                        num+=1
    return mfccs, mu, pi, dic    

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

#r1: taux de données de name utilisé pour l'apprentissage
#r2*|{données de name utilisé pour l'apprentissage}|: nombre de données hors name pour l'apprentissage
#le reste pour les tests
def make_sets(name, dic, r1, r2):
    nameKeys=dic[name]
    advKeys_=np.concatenate([dic[key] for key in dic if key<>name])
    advKeys=advKeys_.tolist()
    l1=len(nameKeys)
    lp1=int(r1*l1)
    l2=len(advKeys)
    lp2=min(int(r2*lp1), l2)
    advs=random.sample(advKeys,lp2)
    for i in advs:
        advKeys.remove(i)
    xApp=nameKeys[0:lp1]+advs
    xTest=nameKeys[lp1:l1]+advKeys
    yApp=[1 for i in xrange(lp1)]+[-1 for i in xrange(lp2)]
    yTest=[1 for i in xrange(l1-lp1)]+[-1 for i in xrange(l2-lp2)]
    return xApp, yApp, xTest, yTest

def train(name,mu0,sig0,mu,pi, xApp, yApp, xTest, yTest,C=1):
    print('learning ' + name)
    # y = build_labels(name,dic)
    # x=range(len(mu))
    #xApp, yApp, xTest, yTest = make_sets(name, dic, r1, r2)
    def k(i,j):
        res = fisher.K(xApp,i,j,mu[i],mu[j],sig0,pi[i],pi[j],mu0)
        return res
    qp(xApp, yApp, xTest, yTest, k, C)
    # w, b = qp(x, y, k, C)
    # return w, b

def predKP(w, b, mu, pi, mu0, sig0, dic, num):
    print 'prediction...'
    yPred=[]
    T=len(w)
    for name in dic:
        for i in dic[name]:
            #calcule <w,i-eme>
            v=[w[j]*fisher.K(w, j, i+num, mu[j], mu[i], sig0, pi[j], pi[j], mu0) for j in xrange(T)]
            tmp=sum(v) + b
            yPred.append(tmp)
            #print name, i+num, ' : ', tmp
    return yPred
            
def evalKP(yPred, nameL, dic, seuil=0):
    ok = 0
    tot = sum(len(val) for val in dic.itervalues())
    j=0
    for name in dic:
        for i in dic[name]:
            if(name==nameL):
                ok+=(yPred[j]-seuil > 0)
            else:
                ok+=(yPred[j]-seuil < 0)
            j+=1
    return ok, ok/float(tot)

def test_make_sets(r1, r2):
    dic={'a':[1, 5, 7, 23, 18, 4, 2],
         'b':[8, 9, 10],
         'c':[6, 11, 12]}
    print dic
    print (make_sets('a', dic, r1, r2))

# test_make_sets(0.5, 1.2)


nbG=4
name='sarkozy'
nbc=100
r1=0.6
r2=1.1

mfccs, mu, pi, dic = createDataFiles(nbc, nbG)

xApp, yApp, xTest, yTest = make_sets(name, dic, r1, r2)
print xApp

print 'GMM sur l\'ensemble des points\n'
mu0, sig0 = gmms([mfccs[i] for i in xApp], nbG)

train(name, mu0, sig0, mu, pi, xApp, yApp, xTest, yTest)
# w, b = train('sarkozy', mu0, sig0, mu, pi, dicApp)

# print 'evaluation...'
# rho = []
# yPred=predKP(w, b, mu, pi, mu0, sig0, dicTest, numApp)
# for i in xrange(3):
#     seuil = 2**(2*(-3+i))
#     k, tmp = evalKP(yPred, 'sarkozy', dicTest, seuil)
#     rho.append(tmp)
# for i in xrange(4):
#     seuil = 2**(3*i)
#     k, tmp = evalKP(yPred, 'sarkozy', dicTest, seuil)
#     rho.append(tmp)

# print rho
