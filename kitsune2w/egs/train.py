import sys
sys.path.append('../model')

import numpy as np
from tqdm import tqdm

from Kitsune2w import Kitsune2w

def train():
    model = Kitsune2w(file_path=None,limit = np.inf, train=True,FM_grace_period = 2000)
    X = np.load("/home/sura/scratch/kitsune2w/data/X.npy")
    y = np.load("/home/sura/scratch/kitsune2w/data/y.npy")
    x = []
    for i in tqdm(range(len(X))):
        ret = model.forward(X[i],y[i])
        if ret == -1:
            break
        
    model.save()


    print("Calculating threshold")
    model.train = False
    model.BenDetector.state_train = False
    model.MalDetector.state_train = False
    rmse_ben,rmse_mal = [],[]
    X = np.load("/home/sura/scratch/kitsune2w/data/weekday.npy")
    for i in tqdm(range(len(X)),desc="thresh"):
        ret = model.forward(X[i],0,False)
        if ret == -1:break
        rmse_ben.append(ret[0])
        rmse_mal.append(ret[1])
    benignSample = np.log(rmse_ben)
    mean, std = np.mean(benignSample), np.std(benignSample)
    threshold_std = np.exp(mean + 3 * std)
    threshold_max = max(rmse_ben)
    threshold = min(threshold_max, threshold_std)
    print("Threshold benign:",threshold)

    benignSample = np.log(rmse_mal)
    mean, std = np.mean(benignSample), np.std(benignSample)
    threshold_std = np.exp(mean + 3 * std)
    threshold_max = max(rmse_mal)
    threshold = min(threshold_max, threshold_std)
    print("Threshold malicious:",threshold)


if __name__ == "__main__":
    train()