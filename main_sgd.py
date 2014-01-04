# -*- coding: utf-8 -*-
# traitement global des fichiers wav
import os,numpy as np,octaveIO as oio,string,subprocess
import sgd
import random
import plot




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

def get3dicts(c,nameInDic,leaveOut,r1,r2,xApp,dicApp,numApp,xTest,dicTest,numTest,xTest2,dicTest2,numTest2):
    if nameInDic not in leaveOut:
        l1,l2,l3 = cutInThree(xrange(len(c)),r1,r2)
    else: # we want to treat people in leaveOut as complete strangers to test against attacks
        l1,l2,l3 = [],[],range(len(c))
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





def createDataFiles(r1,r2,leaveOut,rewrite=True): # r1, r2 and r3 := 1 - r1 - r2 are the ratios of the training, opposite and testing sets
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
                    if rewrite:
                        subprocess.call(['octave', '--silent', '--eval','cepstraux('+'\''+wavFile+'\',\''+mfccFile+'\')'])
                    c=oio.retrieve(mfccFile,['c'])[0] # the frames from the current file
                    l=np.size(c, 0) #nombre total de frames
                    # tempxApp, tempdicApp, tempxTest, tempdicTest, tempxTest2, tempdicTest2 = 
                    numApp,numTest,numTest2 = get3dicts(c,nameInDic,leaveOut,r1,r2,xApp, dicApp, numApp, xTest, dicTest, numTest, xTest2, dicTest2, numTest2)# change Test to Opp and Test2 to Test !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    
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

def trainSgd(name, dic, x,C,iterations=None):
    y = build_labels(name,dic)
    if iterations==None:
        iterations = 10
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

            
def findC(name,dicApp,xApp,dicTest,xTest,n=9,trans=0,iterations=None):
    res = np.zeros((2,n))
    l = [2**(i-(n/2)-trans) for i in xrange(n)]
    for i in xrange(n):
        C = 2**(i-(n/2))
        w2 = trainSgd(name,dicApp,xApp,C,iterations)
        print ('launch sgd for ... '+str(i))
        k, rho = evalSgd(name, w2[:-1], w2[-1], dicTest, xTest)
        res[0,i] = C
        res[1,i] = rho
    l = res[1,:].tolist()
    e = max(l)
    C = res[0,l.index(e)]
    return res,C,e

def makeLegend(leaveOut,r1,r2,iterations,rho,Cfinal):
    res = 'Attackers (i.e. not trained): '
    for i in leaveOut[:-1]:
        res = res + i + ','
    res = res + leaveOut[-1]
    res = res + '\n'
    res = res + str(r1*100) + '% of data used for training'
    res = res + '\n'
    res = res + str(r2*100) + '% of data used for evaluating'
    res = res + '\n'
    res = res + str(iterations) + ' iterations used in the SGD'
    res = res + '\n'
    res = res + 'Result on other data ('+str((1-r1-r2)*100)+'% of total data) from trained speakers \n+ the attackers: '+str(rho)
    res = res + '\n'
    res = res + 'Best value of C: '+str(Cfinal)
    return res

leaveOut = ['antoine','thomas']
r1 = 0.1
r2 = 0.2
iterations=100

print 'create data files...'
xApp, dicApp, xTest, dicTest, xTest2, dicTest2  = createDataFiles(r1,r2,leaveOut,False)
# print 'launch sgd...'
# w2 = trainSgd('sarkozy', dicApp, xApp,10)
# print 'evaluation...'
# # print dicTest2, len(xTest2)
# k, rho = evalSgd('sarkozy', w2[:-1], w2[-1], dicTest, xTest)

# print k, rho

res,C,e = findC('sarkozy',dicApp,xApp,dicTest,xTest,9,-3,iterations)
print res,C,e


print 'launch sgd...'
w2 = trainSgd('sarkozy', dicApp, xApp,C)
k, rho = evalSgd('sarkozy', w2[:-1], w2[-1], dicTest2, xTest2)

print k, rho

# res = np.array([[  0.0625,       0.125,        0.25,         0.5,          1.,           2.,
#     4.,           8.,          16.        ],
#  [  0.75449775,   0.82083958,   0.82083958,   0.78748126,   0.75412294,
#     0.74437781,   0.74512744,   0.74625187,   0.74625187]])

plot.plotFindC(res,makeLegend(leaveOut,r1,r2,iterations,rho,C),e)
