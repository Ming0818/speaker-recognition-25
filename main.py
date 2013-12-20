# -*- coding: utf-8 -*-
# traitement global des fichiers wav
import os,numpy as np,octaveIO as oio,string,subprocess
import fisher
import random
import kernel_perceptron as kp

def createDataFiles(nbc):
    if not os.path.exists('data'):
        os.makedirs('data')
        print "Please add some data, I don't work for free"
    else:
        res = [] #contiendra l'ensemble des mfccs
        muT=[]
        piT=[]
        dic={}
        num=0
        for root, dirs, files in os.walk('data'):
            #print root,dirs,files
            for file in files:
                if file.endswith(".wav"):
                    print "treating file "+file
                    name=os.path.splitext(file)[0]
                    fileName = os.path.join(root, name)
                    wavFile = fileName+'.wav'
                    mfccFile = fileName+'mfcc.mat' #contient 'c'
                    gmmFile = fileName+'gmm.mat' #contient 'mu', 'sig' et 'pi'
                    #print string.join(['octave','--eval','cepstraux('+'\''+wavFile+'\',\''+mfccFile+'\')'])
                    subprocess.call(['octave', '--silent', '--eval','cepstraux('+'\''+wavFile+'\',\''+mfccFile+'\')'])
                    c=oio.retrieve(mfccFile,['c'])[0]
                    l=np.size(c, 0) #nombre total de frames
                    nbv=l/nbc
                    #print nbc, l, nbv
                    dic[name]=[]
                    for j in xrange(min(nbv, 3)):
                        binf=j*nbc
                        bsup=(j+1)*nbc
                        #print 'binf, bsup :', binf, bsup
                        newvf=c[binf:bsup] #vecteur de frames à ajouter
                        #print np.shape(newvf)
                        #print newvf
                        res.append(newvf) #ajout dans l'ensemble complet

                        #Calcul des gmm
                        subprocess.call(['octave','--silent', '--eval','gmm(\''+mfccFile+'\', \''+gmmFile+'\')'])
                        mu, pi = oio.retrieve(gmmFile, ['mu', 'pi'])
                        muT.append(mu)
                        piT.append(pi)
                        dic[name].append(num)
                        num+=1
        return res, muT, piT, dic    

def gmms(data):
    #print data
    #print [np.shape(x) for x in data]
    c0 = np.concatenate(data)
    mfccFile = 'data/mfccs.mat'
    gmmFile = 'data/gmms.mat'
    oio.write(mfccFile,c0,'c')
    #print string.join(['octave','--eval','gmm(\''+mfccFile+'\', \''+gmmFile+'\')'])
    subprocess.call(['octave','--silent', '--eval','gmm(\''+mfccFile+'\', \''+gmmFile+'\')'])
    mu0, sig0=oio.retrieve(gmmFile,['mu','sig'])
    return mu0, sig0

def build_labels(name,dic,mu,pi):
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

def make_training_set(name,dic,m):
    """Take m random adversaries to help train <name>"""
    nameKeys=dic[name]
    advKeys=[dic[key] for key in dic if key<>name]
    advs = random.sample(advKeys,min(len(advKeys),m))
    return {name : nameKeys, '#mechant' : advs}

def train(name,mu0,sig0,mu,pi):
    y = build_labels(name,dic,mu,pi)
    x=range(len(y))
    print 'x:', x
    print 'y:', y
    def k(vi,vj): 
        i=vi[0]
        j=vj[0]
        print i, j
        res = fisher.K(x,i,j,mu[i],mu[j],sig0,pi[i],pi[j],mu0) + vi[1]*vj[1]
        return res
    w = kp.ker_perceptron(x,y,k)
    return w

res, mu, pi, dic = createDataFiles(100)
#print res
print dic
print 'GMM sur l\'ensemble des points\n'

mu0, sig0 = gmms(res)

w = train('thomas', mu0, sig0, mu, pi)
print w
