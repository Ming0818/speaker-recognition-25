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


    # print string.join(['octave', '--silent', '--eval','quad_prog('+'\''+inFile+'\',\''+outFile+'\')'])
    # subprocess.call(['octave', '--silent', '--eval','quad_prog('+'\''+inFile+'\',\''+outFile+'\')'])
    # w = oio.retrieve(outFile, ['w'])
    # return (w[0])[:, 0]

    # cons = [{'type':'eq', 'fun': lambda alp: np.dot(y, alp)}, 
    #         {'type':'ineq', 'fun': lambda alp: alp}, 
    #         {'type':'ineq', 'fun': lambda alp: C - alp}]
    # def objective(x,sign=1.):
    #     return sign*(0.5*np.dot(x.T,np.dot(G,x))- np.dot(e,x))
    # def jacobian(x,sign=1.):
    #     return sign*(np.dot(x.T,G) - e)
    # x0 = np.random.randn(n)
    # res_cons = minimize(objective,x0,jac=jacobian,constraints=cons,method='SLSQP',options={'disp':True})
#    return res_cons
