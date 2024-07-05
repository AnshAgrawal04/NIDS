import sys
sys.path.append("../src/model/kitsune/np")

import numpy as np
np.random.seed(42)
import math

from tqdm import tqdm

from Kitsune import Kitsune

import matplotlib.pyplot as plt


def train():
    # ----------------------------Training---------------------------------
    model = Kitsune(file_path="/home/sura/working_directory/modified_bars/data/train/weekday/weekday.pcap",limit = np.inf, train=True)
    while True:
        ret = model.proc_next_packet()
        if ret == -1:
            break
    model.save()


    print("Calculating threshold")
    model = Kitsune(file_path="/home/sura/working_directory/modified_bars/data/train/weekday/weekday.pcap",limit = np.inf, train=False,load= 0)
    rmse = []
    while True:
        ret = model.proc_next_packet()
        if ret == -1:break
        rmse.append(ret)
    benignSample = np.log(rmse)
    mean, std = np.mean(benignSample), np.std(benignSample)
    threshold_std = np.exp(mean + 3 * std)
    threshold_max = max(rmse)
    threshold = min(threshold_max, threshold_std)
    print("Threshold:",threshold)

def adver_training(version: int = 0):
    # ------------------------Adversarial Training----------------------------
    model = Kitsune(file_path=None,limit = np.inf, train=True,load= version)
    x_adv = np.load("/home/sura/working_directory/modified_bars/data/eval/service_detection/X_test.npy")
    print(x_adv.shape)
    labels = np.load("/home/sura/working_directory/modified_bars/data/eval/service_detection/y_test.npy")
    epochs = 100
    for epoch in range(epochs):
        for i in tqdm(range(len(x_adv)),desc=str(epoch)):
            model.forward(x_adv[i],labels[i],adver=True)
    model.save()

    print("Calculating threshold")
    model = Kitsune(file_path="/home/sura/working_directory/modified_bars/data/train/weekday/weekday.pcap",limit = np.inf, train=False,load= version+1)
    rmse = []
    while True:
        ret = model.proc_next_packet()
        if ret == -1:break
        rmse.append(ret)
    benignSample = np.log(rmse)
    mean, std = np.mean(benignSample), np.std(benignSample)
    threshold_std = np.exp(mean + 3 * std)
    threshold_max = max(rmse)
    threshold = min(threshold_max, threshold_std)
    print("Threshold:",threshold)

def eval(version:int = 0):
# ----------------------------Evaluation-------------------------------
    # file_name="Smartphone_1_ACK_Flooding_iter_0"
    # model = Kitsune(file_path="/home/sura/scratch/Adversarial_Data/adversarial/"+file_name+".pcap",limit = np.inf, train=False,load= version)
    model = Kitsune(file_path="/home/sura/scratch/UQIOT/69897e94e24170c0_UQIOT2022_A7369/data/attack_samples/Lenovo_Bulb_1/ACK_Flooding_Lenovo_Bulb_1.pcap",limit = np.inf, train=False,load= version)
    rmse = []
    ad_threshold = 0.23129002911825286
    acc = 0
    count = 0
    while True:
        print("\rPackets Evaluated: ",count,flush=True,end="")
        ret = model.proc_next_packet()
        if ret == -1:break
        rmse.append(ret)
        if(ret>ad_threshold):acc+=1
        count+=1
    import matplotlib.pyplot as plt
    _, ax1 = plt.subplots(constrained_layout=True, figsize=(10, 5), dpi=200)
    ax1.axhline(y=ad_threshold, color='r', linestyle='-')
    x_val = np.arange(len(rmse))
    print(file_name)
    print("\nAccuracy: "+str(acc/len(rmse)*100))
    ax1.scatter(x_val, rmse, s=1, c='#00008B')
    ax1.set_yscale("log")
    ax1.set_ylabel("RMSE (log scaled)")
    ax1.set_xlabel("packet index")
    plt.legend([ "Threshold","Anomaly Score"])
    plt.savefig("figures/anomaly_mal_ack_lenovo.png")

def test(constant_,train_size):
    acc = []
    thresh_arr = [0.23129002911825286]
    x_adv = np.load("/home/sura/scratch/dataset_manager/X_test.npy")
    labels = np.load("/home/sura/scratch/dataset_manager/y_test.npy")
    x_thresh = np.load("/home/sura/working_directory/modified_bars/data/train/complete_weekday/X_train.npy")
    x_thresh = x_thresh[:train_size]
    data = np.load("/home/sura/scratch/adversarial_training/data/test.npy")
    #data =np.load("/home/sura/scratch/dataset_manager/results/Lenovo_Bulb_1_ACK_Flooding_iter_0.npy")
    ind = np.random.choice(range(0, len(data)-1), size=1000, replace=False)
    data  = data[ind]
    model = Kitsune(file_path=None,limit = np.inf, train=False,load= 0 )
    c,t=0,0
    for x in data:
        ret = model.forward(x,0,False)
        if ret == -1:break
        if ret > thresh_arr[-1]:
            c+=1
        else:
            pass
        t+=1
    acc.append(c/t)
    print(acc)
    for epoch in tqdm(range(1,101)):
        model.train = True
        model.AnomDetector.state_train = True
        for i in range(len(x_adv)):
            # model.forward(x_adv[i],np.log(constant_*thresh_arr[epoch-1]*labels[i]+1),adver=True)
            # model.forward(x_adv[i], np.maximum(0, np.tanh(constant_ * thresh_arr[epoch-1] * labels[i])), adver=True)
            # model.forward(x_adv[i],np.maximum(0,np.log(1+np.exp(constant_*thresh_arr[epoch-1]*labels[i]))),adver=True)
            model.forward(x_adv[i],constant_*thresh_arr[epoch-1]*labels[i],adver=True)
        model.train = False
        model.AnomDetector.state_train = False
        rmse = []
        for x in x_thresh:
            ret = model.forward(x,0,False)
            if ret == -1:break
            rmse.append(ret)
        benignSample = np.log(rmse)
        mean, std = np.mean(benignSample), np.std(benignSample)
        threshold_std = np.exp(mean + 3 * std)
        threshold_max = max(rmse)
        threshold = min(threshold_max, threshold_std)
        thresh_arr.append(threshold)
        threshold = thresh_arr[0]
        c,t=0,0
        for x in data:
            ret = model.forward(x,0,False)
            if ret == -1:break
            if ret > threshold:
                c+=1
            else:
                pass
            t+=1
        acc.append(c/t)
    print(acc)
    print(thresh_arr)
    # changes from here
    plt.clf()
    plt.plot(acc)
    plt.savefig("graphs/results_"+str(train_size)+"/Curr "+str(constant_)+".png")


if __name__ == "__main__":
    # train()
    # adver_training()
    # eval()
    test(10,10)
    


