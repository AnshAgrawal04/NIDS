import os 

import torch
import torch.nn as nn

from autoencoder import Autoencoder
from detect import Detector
from preprocess import Normalizer, Feature_Map

import yaml
config_file_path = 'config/torch_adv_config.yaml'
with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)

n_gpu = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() and not config["gpu"]["no_gpu"] else "cpu")

save_dir_train = "/home/sura/working_directory/adversarial_training/src/model/kitsune/pytorch/checkpoints"

def get_threshold():
    if(os.path.exists(save_dir_train+"/threshold")):
        with open(save_dir_train+"/threshold","r") as f:
            ad_threshold = float(f.read()) # Threshold for anomaly detection
    else:ad_threshold = 0.1
    return ad_threshold

def create_ae(feature_size: int) -> nn.Module:
    return Autoencoder(feature_size, device)

def create_normalizer():
    return Normalizer(device)

def create_fm():
    return Feature_Map(device)

def create_detector(normalizer: nn.Module, fm: nn.Module, ae: nn.Module) -> nn.Module:
    return Detector(normalizer, fm, ae, get_threshold(), device)

def create_comp_detector() -> nn.Module:
    normalizer = create_normalizer()
    norm_param = torch.load(os.path.join(save_dir_train, "checkpoint-norm"))
    normalizer.norm_max = norm_param["norm_max"].to(device)
    normalizer.norm_min = norm_param["norm_min"].to(device)    
    fm = create_fm()
    fm.mp = torch.load(os.path.join(save_dir_train, "checkpoint-fm"))
    detector = create_detector(normalizer, fm, nn.ModuleList([torch.load(os.path.join(save_dir_train, "checkpoint-ae-%d" % (i)
        )).to(device) for i in range(fm.get_num_clusters())]))
    return detector