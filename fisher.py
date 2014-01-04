# -*- coding: utf-8 -*-
# calcul du noyau de Fisher (avec l'identité pour l'instant)
import octaveIO as oio
import numpy as np


def nim(m,pi,T):
    res = T*(pi[m]) 
    return res

def K(x,i,j,mu_i,mu_j,sig0,pi_i,pi_j,mu0):
    T = len(x)
    #print mu0
    (M,F) = np.shape(mu_i) # M: nombre de Gaussiennes. F: nombre de features
    res = 0
    ni = np.zeros(M)
    nj = np.zeros(M)
    for m in xrange(M):
        ni[m]=nim(m,pi_i,T)
        nj[m]=nim(m,pi_j,T)
        ti = np.array([(mu_i[m, k] - mu0[m,k])/(sig0[m][k, k]**2) for k in xrange(F)])
        tj = np.array([(mu_j[m, k] - mu0[m,k])/(sig0[m][k, k]**2) for k in xrange(F)])
        res += ni[m]*nj[m]*np.dot(ti,tj)
    return res
        
