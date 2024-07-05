import sys
sys.path.append('../model')

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from Kitsune2w import Kitsune2w

def roc():
    model = Kitsune2w(file_path="/home/sura/scratch/kitsune2w/data/weekday.npy",limit = np.inf, train = False,load = 0 )
    y1,y2,y=[],[],[]
    while True:
        res = model.proc_next_packet()
        if(res==-1):break
        y1.append(res[0])
        y2.append(res[1])
        y.append(0)
    X = np.load("/home/sura/scratch/kitsune2w/data/malicious/udp.npy")
    for i in range(len(X)):
        res = model.forward(X[i])
        y1.append(res[0])
        y2.append(res[1])
        y.append(1)
    X = np.load("/home/sura/scratch/kitsune2w/data/malicious/ack.npy")
    for i in range(len(X)):
        res = model.forward(X[i])
        y1.append(res[0])
        y2.append(res[1])
        y.append(1)
    y1 = np.array(y1)
    y2 = -np.array(y2)
    y = np.array(y)
    fpr1, tpr1, _ = roc_curve(y, y1)
    roc_auc1 = auc(fpr1, tpr1)

    fpr2, tpr2, _ = roc_curve(y, y2)
    roc_auc2 = auc(fpr2, tpr2)

    # Plot ROC curves
    plt.figure()
    plt.plot(fpr1, tpr1, color='blue', lw=2, label='Model 1 ROC curve (area = %0.2f)' % roc_auc1)
    plt.plot(fpr2, tpr2, color='red', lw=2, label='Model 2 ROC curve (area = %0.2f)' % roc_auc2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("roc.png")

def eval(file_path,save_name):
    model = Kitsune2w(file_path= file_path,limit = int(1e5),train = False, load = 0)
    ben ,mal =[],[]
    ret = model.proc_next_packet()
    for i in range(10000):
        if(ret==-1):break
        ben.append(ret[0])
        mal.append(ret[1])
        ret = model.proc_next_packet()
    x_val = np.arange(len(ben))
    _, ax1 = plt.subplots(constrained_layout=True, figsize=(10, 5), dpi=200)
    ax1.scatter(x_val, ben, s=1, c='#00008B')
    ax1.scatter(x_val, mal, s=1, c='r')
    ax1.set_yscale("log")
    ax1.set_ylabel("RMSE (log scaled)")
    ax1.set_xlabel("packet index")
    plt.legend([ "Threshold","Anomaly Score"])
    plt.savefig("results/adversarial/"+save_name+".png")

def mean_std_dif(file_path):
    model = Kitsune2w(file_path= file_path,limit = int(1e5),train = False, load = 0)
    data =[]
    c2=0
    ret = model.proc_next_packet()
    for i in range(10000):
        if(ret==-1):break
        if(ret[0]<0.19):c2+=1
        data.append(ret[1]-ret[0])
        ret = model.proc_next_packet()
    data = np.array(data)
    count = sum(data>=-0.08)
    print("Accuracy:", count/len(data))
    print("Mean:",np.mean(data),"\nStd:",np.std(data) )

def threshold():
    model = Kitsune2w(file_path= "/home/sura/scratch/kitsune2w/data/malicious/udp.npy",limit = int(1e5),train = False, load = 0)
    data =[]
    c2=0
    ret = model.proc_next_packet()
    for i in range(10000):
        if(ret==-1):break
        if(ret[0]<0.19):c2+=1
        data.append(ret[1]-ret[0])
        ret = model.proc_next_packet()
    data = np.array(data)
    acc = []
    for i in range(-50,11,5):
        acc.append(sum(data<i/100)/len(data))
    print(acc)

mean_std_dif("/home/sura/scratch/kitsune2w/data/weekday.npy")