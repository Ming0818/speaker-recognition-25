# -*- coding: utf-8 -*-
# traitement global des fichiers wav
import os,numpy as np,octaveIO as oio,string,subprocess
import sgd
import random





# def chooseTrainingData(name,dic,ratio):
#     size = sum(len(val) for val in dic.itervalues())
#     sFriend = len(dic[name]) #number of samples for our training candidate
#     sFoe = size - sFriend # number of other samples
#     keepFriend = random.sample(dic[name],int(ratio*sFriend))
#     OutFriend = [x for x in dic[name] if x not in keepFriend]
#     dicKeep = {}
#     dicKeep[name] = keepFriend
#     dicOut = {}
#     for foe in dic:
#         if foe<>name:


def cutInThree(l,ratio1,ratio2): # and of course ratio3 = 1 - ratio1 - ratio2
    s1 = int(ratio1*len(l))
    l1 = [l[i] for i in xrange(s1)]
    s2 = int(ratio2*len(l))
    l2 = [l[i+s1] for i in xrange(s2)]
    l3 = list((set(l) - set(l1) - set(l2)))
    # l1 = random.sample(l,int(ratio1*len(l)))
    # lRem = list(set(l) - set(l1))
    # l2 = random.sample(lRem,int(ratio2*len(l1)))
    # l3 = list(set(lRem) - set(l2))
    return l1,l2,l3

def get3dicts(c,nameInDic,r1,r2,xApp,dicApp,numApp,xTest,dicTest,numTest,xTest2,dicTest2,numTest2):
    l1,l2,l3 = cutInThree(xrange(len(c)),r1,r2)
    if(not dicApp.has_key(nameInDic)):
        dicApp[nameInDic]=[]
        dicTest[nameInDic]=[]
        dicTest2[nameInDic]=[]
    for j in l1:
        xApp.append(c[j])#.tolist()
        dicApp[nameInDic].append(numApp)
        numApp += 1
    for j in l2:
        xTest.append(c[j])#.tolist()
        dicTest[nameInDic].append(numTest)
        numTest += 1
    for j in l3:
        xTest2.append(c[j])#.tolist()
        dicTest2[nameInDic].append(numTest2)
        numTest2 += 1
    return numApp,numTest,numTest2
    # return xApp, dicApp, xTest, dicTest, xTest2, dicTest2





def createDataFiles(r1,r2): # r1, r2 and r3 := 1 - r1 - r2 are the ratios of the training, opposite and testing sets
    xApp = [] #contiendra l'ensemble des mfccs à apprendre
    dicApp={}
    numApp=0
    xTest=[]
    dicTest={}
    numTest=0 
    xTest2=[]
    dicTest2={}
    numTest2=0 
    if not os.path.exists('data'):
        os.makedirs('data')
        print "Please add some data, I don't work for free"
    else:
        for root, dirs, files in os.walk('data'):
            for file in files:
                if file.endswith(".wav"):
                    print "treating file "+file
                    print root
                    nameInDic=os.path.split(root)[-1]
                    name=os.path.splitext(file)[0]
                    fileName = os.path.join(root, name)
                    wavFile = fileName+'.wav'
                    mfccFile = fileName+'mfcc.mat' #contient 'c'
                    subprocess.call(['octave', '--silent', '--eval','cepstraux('+'\''+wavFile+'\',\''+mfccFile+'\')'])
                    c=oio.retrieve(mfccFile,['c'])[0] # the frames from the current file
                    l=np.size(c, 0) #nombre total de frames
                    # tempxApp, tempdicApp, tempxTest, tempdicTest, tempxTest2, tempdicTest2 = 
                    numApp,numTest,numTest2 = get3dicts(c,nameInDic,r1,r2,xApp, dicApp, numApp, xTest, dicTest, numTest, xTest2, dicTest2, numTest2)# change Test to Opp and Test2 to Test !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    
        return xApp, dicApp, xTest, dicTest, xTest2, dicTest2

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

def trainSgd(name, dic, x,C):
    y = build_labels(name,dic)
    iterations = 100
    w = sgd.sgd(x,y,np.zeros(len(x[0])+1),iterations,1,sgd.L,0.01,C)
    return w

def evalSgd(nameL, w, b, dic, x):
    tot = len(x)
    ok = 0
    for name in dic:
        for i in dic[name]:
            # print len(x),i
            tmp = np.dot(w, x[i]) + b
            if (name == nameL):
                ok += (tmp > 0)
            else:
                ok += (tmp < 0)
            #print name, i, ' : ', tmp
    return ok, ok/float(tot)

            
def findC(name,dicApp,xApp,dicTest,xTest,n=9):
    res = np.zeros((2,n))
    l = [2**(i-(n/2)) for i in xrange(n)]
    for i in xrange(n):
        C = 2**(i-(n/2))
        w2 = trainSgd(name,dicApp,xApp,C)
        print ('launch sgd for ... '+str(i))
        k, rho = evalSgd(name, w2[:-1], w2[-1], dicTest, xTest)
        res[0,i] = C
        res[1,i] = rho
    l = res[1,:].tolist()
    e = max(l)
    C = l.index(e)
    return res,C

print 'create data files...'
xApp, dicApp, xTest, dicTest, xTest2, dicTest2  = createDataFiles(0.2,0.5)
# print 'launch sgd...'
# w2 = trainSgd('sarkozy', dicApp, xApp,10)
# print 'evaluation...'
# # print dicTest2, len(xTest2)
# k, rho = evalSgd('sarkozy', w2[:-1], w2[-1], dicTest, xTest)

# print k, rho

res,C = findC('sarkozy',dicApp,xApp,dicTest,xTest,4)
print res,C


print 'launch sgd...'
w2 = trainSgd('sarkozy', dicApp, xApp,C)
k, rho = evalSgd('sarkozy', w2[:-1], w2[-1], dicTest2, xTest2)

print k, rho
