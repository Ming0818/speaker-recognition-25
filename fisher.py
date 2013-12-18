# -*- coding: utf-8 -*-
# calcul du noyau de Fisher (avec l'identité pour l'instant)
import octaveIO as oio


def nim(m,pi,T):
    res = T*pi[m] #hypothèse à la lecture de l'article + http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/doc/voicebox/gaussmix.html

def K(x,i,j,mu_i,mu_j,sig0,pi,mu0):
    T = len(x)
    (M,F) = np.shape(mu) # M: nombre de Gaussiennes. F: nombre de features
    res = 0
    ni = np.zeros(M)
    nj = np.zeros(M)
    for m in xrange(M):
        ni[m]=nim(m,pi,T)
        nj[m]=nim(m,pi,T)
        ti = np.array([(mu_i[m,k] - mu0[k])/(sig0[k]^2) for k in xrange(F)])
        tj = np.array([(mu_j[m,k] - mu0[k])/(sig0[k]^2) for k in xrange(F)])
        res += ni[m]*nj[m]*np.dot(ti,tj)
    return res
        

# def Fisher()