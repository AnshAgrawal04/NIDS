import numpy as np

from tqdm import tqdm

import sys
sys.path.append("./fe/")
from FeatureExtractor import FE
pcap_files=[]
# pcap_file = "/home/sura/scratch/adversarial_training/data/Raspberry_Pi_telnet_UDP_Flooding_iter_0.pcap"
# pcap_files.append("/home/sura/scratch/Adversarial_Data/adversarial/Google-Nest-Mini_1_UDP_Flooding_iter_0.pcap")
# pcap_files.append("/home/sura/scratch/Adversarial_Data/adversarial/Lenovo_Bulb_1_UDP_Flooding_iter_0.pcap")
pcap_files.append("/home/sura/scratch/Adversarial_Data/adversarial/Raspberry_Pi_telnet_UDP_Flooding_iter_0.pcap")
# pcap_files.append("/home/sura/scratch/Adversarial_Data/adversarial/Smartphone_1_UDP_Flooding_iter_0.pcap")
# pcap_files.append("/home/sura/scratch/Adversarial_Data/adversarial/Smart_Clock_1_UDP_Flooding_iter_0.pcap")
# pcap_files.append("/home/sura/scratch/Adversarial_Data/adversarial/Lenovo_Bulb_1_ACK_Flooding_iter_0.pcap")
for pcap_file in pcap_files:
    limit = int(1e5)

    fe = FE(pcap_file,limit)
    limit = fe.get_limit()
    data = []

    for i in tqdm(range(limit),desc="convert_to_npy"):
        x = fe.get_next_vector()
        if(len(x)==0):
            break
        data.append(x)
    data = np.array(data)
    pcap_file = pcap_file.split("/")[-1].split(".")[0]
    np.save("./results/"+pcap_file+".npy",data)
