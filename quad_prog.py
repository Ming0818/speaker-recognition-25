import numpy as np
import octaveIO as oio
import subprocess, string
from scipy.optimize import minimize, rosen, rosen_der

def qp(xApp, yApp, xTest, yTest, k, C):
    n = len(xApp)
    m=len(xTest)
    G = np.zeros((n, n))
    K = np.zeros((n+m,n))
    for i in xrange(n):
        for j in xrange(i):
            tmp = k(xApp[i], xApp[j])
            G[i,j] = G[j,i] = yApp[i]*yApp[j]*tmp
            K[i, j]=K[j, i]=tmp
        K[i, i]=G[i,i] = k(xApp[i], xApp[i])
        for j in xrange(n, n+m):
            K[j,i]=k(xApp[i],xTest[j-n])
            

    inFile='data/qp_data.mat'
    outFile='data/qp_res.mat'
    oio.writeDic(inFile, {'G': G, 'K': K, 'yApp':yApp, 'yTest':yTest, 'C': C})

    subprocess.call(['matlab', '-nojvm', '-r', 'qp('+'\''+inFile+'\',\''+outFile+'\')'])
    return (oio.retrieve(outFile, ['w', 'b', 'accTest']))
    
