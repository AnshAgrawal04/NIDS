import os, random
import numpy as np
import sys
sys.path.append('../src/model/kitsune/pytorch')


from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import seaborn as sns
sns.set_style("darkgrid")

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score

from utils import load_data, AverageMeter, calc_metrics_classifier, calc_metrics_certify

import create

import yaml
config_file_path = 'config/torch_adv_config.yaml'
with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)

n_gpu = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() and not config["gpu"]["no_gpu"] else "cpu")

seed =42

save_dir_train = "/home/sura/working_directory/adversarial_training/src/model/kitsune/pytorch/checkpoints"

def set_seed():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def get_threshold():
    print("\n***** Run threshold *****")

    detector = create.create_comp_detector()
    x_train = torch.tensor(np.load("/home/sura/working_directory/modified_bars/data/train/weekday/X_train.npy"), dtype=torch.float)
    dataset = TensorDataset(x_train)
    data_loader = DataLoader(dataset, batch_size=config["classifier"]["batch_size_train"], shuffle=False)
    rmse_array = np.array([])
    for (X,) in tqdm(data_loader,desc="Calculating threshold"):
        rmse_array=np.append(rmse_array,detector.score(X.to(device)).detach().cpu().numpy())
    benignSample = np.log(rmse_array)
    mean, std = np.mean(benignSample), np.std(benignSample)
    threshold_std = np.exp(mean + 3 * std)
    threshold_max = max(rmse_array)
    threshold = min(threshold_max, threshold_std)
    print("Threshold:",threshold)
    with open(os.path.join(save_dir_train,"threshold"),"w") as f:
        f.write(str(threshold))
    
def train():
    print("\n***** Run clustering *****")

    x_cluster = torch.tensor(np.load("/home/sura/working_directory/modified_bars/data/train/weekday/X_cluster.npy"), dtype=torch.float)
    fm = create.create_fm()
    fm.init(x_cluster)

    torch.save(fm.mp, os.path.join(save_dir_train, "checkpoint-fm"))

    criterion = nn.MSELoss()

    x_train = torch.tensor(np.load("/home/sura/working_directory/modified_bars/data/train/weekday/X_train.npy"), dtype=torch.float)
    dataset = TensorDataset(x_train)
    data_loader = DataLoader(dataset, batch_size=config["classifier"]["batch_size_train"], shuffle=False)

    normalizer = create.create_normalizer()
    normalizer.update(x_train) 
    torch.save({"norm_max": normalizer.norm_max, "norm_min": normalizer.norm_min
        }, os.path.join(save_dir_train, "checkpoint-norm"))

    for i in range(fm.get_num_clusters()):
        print("\n***** Run training AE %d *****" % (i))

        ae = create.create_ae(len(fm.mp[i]))
        ae.to(device)

        opt = optim.Adam(ae.parameters(), lr=float(config["classifier"]["learning_rate_classifier"]))

        ae.train()
        for epoch in range(config["classifier"]["num_epochs_classifier"] + 1):
            loss_record = AverageMeter()
            for (X,) in data_loader:
                X = X.to(ae.device)

                X = normalizer(X)
                X = X[:, fm.mp[i]]
                x_reconstructed = ae(X)
                loss = criterion(X, x_reconstructed)

                if epoch > 0:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                loss_record.update(loss.item())

            if epoch % config["classifier"]["print_step_classifier"] == 0:
                print(('Epoch: [%d/%d] | MSE Loss (Avg): %.6f') % ( \
                    epoch, config["classifier"]["num_epochs_classifier"], loss_record.avg))

        torch.save(ae.ae, os.path.join(save_dir_train, "checkpoint-ae-%d" % (i)))
    get_threshold()

    

def evaluate():
    print("\n***** Run evaluating *****")

    model = create.create_comp_detector()
    x_test = torch.tensor(np.load("/home/sura/working_directory/modified_bars/data/eval/arp_spoofing/X_test.npy"),dtype = torch.float)
    y_test = torch.tensor(np.load("/home/sura/working_directory/modified_bars/data/eval/arp_spoofing/y_test.npy"),dtype = torch.float)

    dataset = TensorDataset(x_test, y_test)
    data_loader = DataLoader(dataset, batch_size=config["classifier"]["batch_size_eval"], shuffle=False)
    rmse_record = np.array([], dtype=np.float32)
    pred_record = np.array([], dtype=np.long)
    label_record = np.array([], dtype=np.long)

    torch.set_grad_enabled(False)
    model.eval()
    for X, y in tqdm(data_loader, desc="Evaluate"):
        X, y = X.to(model.device), y.to(model.device)

        pred = model(X)
        rmse = model.score(X)

        rmse_record = np.concatenate([rmse_record, rmse.detach().cpu().numpy()], 0)
        pred_record = np.concatenate([pred_record, pred.detach().cpu().numpy()], 0)

        label_record = np.concatenate([label_record, y.detach().cpu().numpy()], 0)
    torch.set_grad_enabled(True)
    torch.cuda.empty_cache()
    acc, p, r, f1 = np.array((label_record == pred_record), dtype=float).mean(),precision_score(label_record, pred_record),recall_score(label_record, pred_record),f1_score(label_record, pred_record)
    print(f"Accuracy: {acc:.4f}, Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
    _, ax1 = plt.subplots(constrained_layout=True, figsize=(10, 5), dpi=200)
    x_val = np.arange(len(rmse_record))
    ax1.scatter(x_val, rmse_record, s=1, c='#00008B')
    ax1.axhline(y=model.ad_threshold, color='r', linestyle='-')
    ax1.set_yscale("log")
    ax1.set_ylabel("RMSE (log scaled)")
    ax1.set_xlabel("packet index")
    plt.legend([ "Threshold","Anomaly Score"])
    plt.savefig("anomaly_score.png")
    plt.close()

def adversarial_training():

if __name__ == '__main__':
    set_seed()
    # train()
    evaluate()