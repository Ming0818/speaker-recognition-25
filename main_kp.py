# -*- coding: utf-8 -*-
# traitement global des fichiers wav
import os,numpy as np,octaveIO as oio,string,subprocess
import fisher
import random
from sklearn import mixture
from kernel_perceptron import kp
import pickle

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
                    # print "treating file "+file
                    nameInDic=os.path.split(root)[-1]
                    # print "-> "+nameInDic
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
#leaveOut est une liste de noms de locuteurs éventuels à ne pas inclure dans le training
def make_sets(name, dic, r1, r2,leaveOut=[]):
    nameKeys=dic[name]
    advKeys_=np.concatenate([dic[key] for key in dic if (key<>name and key not in leaveOut)])
    if leaveOut:
        advKeysLeaveOut_ = np.concatenate([dic[key] for key in dic if (key<>name and key in leaveOut)])
        advKeysLeaveOut=advKeysLeaveOut_.tolist()
    else:
        advKeysLeaveOut = []
    advKeys=advKeys_.tolist()    
    l1=len(nameKeys)
    lp1=int(r1*l1)
    l2=len(advKeys)
    lp2=min(int(r2*lp1), l2)
    advs=random.sample(advKeys,lp2)
    for i in advs:
        advKeys.remove(i)
    testLengthAdv = l2/2 # on ne teste que sur la moitié de ce qui reste pour pouvoir faire un "blind" test sur un troisième ensemble, xTest2
    testLengthName = l1/2
    xApp=nameKeys[0:lp1]+advs
    xTest=nameKeys[lp1:testLengthName]+advKeys[0:testLengthAdv]
    xTestPlus=testLengthName-lp1
    xTestMinus=testLengthAdv
    xTest2 = nameKeys[(testLengthName+1):l1]+advKeys[(testLengthAdv+1):]+advKeysLeaveOut
    xTest2Plus=(l1-testLengthName-1)
    xTest2Minus=(len(advKeys)-1 -testLengthAdv)+len(advKeysLeaveOut)
    yApp=[1 for i in xrange(lp1)]+[-1 for i in xrange(lp2)]
    yTest=[1 for i in xrange(xTestPlus)]+[-1 for i in xrange(xTestMinus)]
    yTest2=[1 for i in xrange(xTest2Plus)]+[-1 for i in xrange(xTest2Minus)]
    return xApp, yApp, xTest, yTest,xTest2,yTest2

def train(name,mu0,sig0,mu,pi, xApp, yApp, verbose=False, stop=100000):
    #print('learning ' + name)
    # y = build_labels(name,dic)
    # x=range(len(mu))
    #xApp, yApp, xTest, yTest = make_sets(name, dic, r1, r2)
    def k(i,j):
        res = fisher.K(xApp,i,j,mu[i],mu[j],sig0,pi[i],pi[j],mu0)
        return res
    return kp(xApp, yApp, k, verbose=verbose, stop=stop)

def predKP(w, b, mu, pi, mu0, sig0, xTest):
    # print 'prediction...'
    yPred=[]
    T=len(w)
    for i in xTest:
            #calcule <w,i-eme>
        v=[w[j]*fisher.K(w, j, i, mu[j], mu[i], sig0, pi[j], pi[j], mu0) for j in xrange(T)]
        tmp=sum(v) + b
        yPred.append(tmp)
    return yPred
            
def evalKP(yPred, yTest):
    ok = 0
    tot = len(yTest)
    for i in xrange(tot):
        ok+=(yPred[i]*yTest[i] > 0)
    return ok/float(tot)

def test_make_sets(r1, r2,leaveOut=[]):
    dic={'a':[1, 5, 7, 23, 18, 4, 2],
         'b':[8, 9, 10],
         'c':[6, 11, 12]}
    # print dic
    res = make_sets('a', dic, r1, r2,leaveOut)
    return dic,res

#dic, res = test_make_sets(0.2, 1.2,['b'])
#print dic,res

nbc=100

# nbG=50
# name='gerra'
# r1=0.6
# r2=2.0

# mfccs, mu, pi, dic = createDataFiles(nbc, nbG)
# print name
# xApp, yApp, xOpp, yOpp, xTest, yTest = make_sets(name, dic, r1, r2)

# print 'GMM sur l\'ensemble des points\n'
# mu0, sig0 = gmms([mfccs[i] for i in xApp], nbG)
# print 'start training'
# w, b, ind = train(name, mu0, sig0, mu, pi, xApp, yApp, verbose=True)
# acc = evalKP(predKP(w, b, mu, pi, mu0, sig0, xTest), yTest)
# print ind, acc

def optimisation(names, nbGs, r1, r2, stop=100000):
    res={}
    for nbG in nbGs:
        mfccs, mu, pi, dic = createDataFiles(nbc, nbG)
        pickle.dump(dic, open('data_kp'+str(nbG)+'.dat', 'wb'))
        for name in names:
            xApp, yApp, xOpp, yOpp, xTest, yTest = make_sets(name, dic, r1, r2)
            mu0, sig0 = gmms([mfccs[i] for i in xApp], nbG)
            w, b, it = train(name, mu0, sig0, mu, pi, xApp, yApp, stop=stop)
            acc = evalKP(predKP(w, b, mu, pi, mu0, sig0, xTest), yTest)
            res[(name, nbG)]=acc
            print name, nbG, acc, it
    return res

res = optimisation(['gerra', 'sarkozy', 'L4', 'thomas'], [10, 30, 40, 50, 70, 100], 0.6, 2.0, stop=300)

pickle.dump(res, open('resultats_kp.dat', 'wb'))

