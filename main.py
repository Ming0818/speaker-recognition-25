# -*- coding: utf-8 -*-
# traitement global des fichiers wav
import os,numpy,octaveIO,string,subprocess



def createDataFiles():
    if not os.path.exists('data'):
        os.makedirs('data')
        print "Please add some data, I don't work for free"
    else:
        res = []
        for root, dirs, files in os.walk('data'):
            print root,dirs,files
            for file in files:
                if file.endswith(".wav"):
                    print "treating file "+file
                    name=os.path.splitext(file)[0]
                    fileName = os.path.join(root, name)
                    wavName = fileName+'.wav'
                    matName = fileName+'.mat'
                    #print string.join(['octave','--eval','cepstraux('+'\''+wavName+'\',\''+matName+'\')'])
                    subprocess.call(['octave','--eval','cepstraux('+'\''+wavName+'\',\''+matName+'\')'])
                    triplet=octaveIO.retrieve(matName,['mu','sig','pi'])
                    res.append(triplet)
        return res
                    

res = createDataFiles()
print res
