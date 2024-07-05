import sys
sys.path.append('../model')

import numpy as np
from tqdm import tqdm

from Kitsune2w import Kitsune2w

def adv_train():
    model = Kitsune2w(file_path=None,limit = np.inf, train=True,FM_grace_period = 2000,load = 0)
    X = np.load("/home/sura/scratch/kitsune2w/data/X_adv.npy")
    y = np.load("/home/sura/scratch/kitsune2w/data/y_adv.npy")
    X_eval = [np.load("/home/sura/scratch/kitsune2w/data/adv/cam_ack.npy")]
    y_eval = [np.ones(len(X_eval[0]))]
    epochs = 10
    acc = [[],[]]
    for i in tqdm(range(100)):
        model.train = True
        model.BenDetector.state_train = True
        model.MalDetector.state_train = True
        for i in range(len(X)):
            ret = model.forward(X[i],y[i])
            if ret == -1:
                break
        c2=0
        c1 = 0
        model.train = False
        model.BenDetector.state_train = False
        model.MalDetector.state_train = False
        for t in range(len(X_eval)):
            for i in range(10000):
                ret = model.forward(X_eval[t][i],y_eval[t][i])
                if(ret==-1):break
                if((y_eval[t][i]==1)!=(ret[1]-ret[0]>-0.08)):c1+=1
                c2+=1
            acc[t].append(c1/c2)
    print(acc)

adv_train()