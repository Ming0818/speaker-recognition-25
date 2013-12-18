# -*- coding: utf-8 -*-
# fichier gérant l'interface avec octave

import scipy.io

def retrieve(matFile,matNames):
    mat = scipy.io.loadmat(matFile)
    res = [mat[matName] for matName in matNames]
    return res


