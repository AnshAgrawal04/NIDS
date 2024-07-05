import numpy as np

np.random.seed(20)

ben = ["/home/sura/working_directory/modified_bars/data/train/complete_weekday/X_train.npy"]
mal = ["/home/sura/working_directory/modified_bars/data/eval/ack_flooding/X_test.npy","/home/sura/working_directory/modified_bars/data/eval/arp_spoofing/X_test.npy","/home/sura/working_directory/modified_bars/data/eval/port_scanning/X_test.npy","/home/sura/working_directory/modified_bars/data/eval/service_detection/X_test.npy","/home/sura/working_directory/modified_bars/data/eval/ssdp/X_test.npy","/home/sura/working_directory/modified_bars/data/eval/syn_flooding/X_test.npy","/home/sura/working_directory/modified_bars/data/eval/udp_flooding/X_test.npy"]
mal=["/home/sura/scratch/dataset_manager/results/Google-Nest-Mini_1_UDP_Flooding_iter_0.npy","/home/sura/scratch/dataset_manager/results/Lenovo_Bulb_1_UDP_Flooding_iter_0.npy","/home/sura/scratch/dataset_manager/results/Raspberry_Pi_telnet_UDP_Flooding_iter_0.npy","/home/sura/scratch/dataset_manager/results/Smartphone_1_UDP_Flooding_iter_0.npy","/home/sura/scratch/dataset_manager/results/Smart_Clock_1_UDP_Flooding_iter_0.npy","/home/sura/scratch/dataset_manager/results/SmartTV_UDP_Flooding_iter_0.npy"]
def generate_list_sum_to_n(n, length):
    random_numbers = np.random.randint(1, n // 2, length - 1)
    current_sum = np.sum(random_numbers)
    last_number = n - current_sum
    while last_number <= 0 :
        random_numbers = np.random.randint(1, n // 2, length - 1)
        current_sum = np.sum(random_numbers)
        last_number = n - current_sum
    result = np.append(random_numbers, last_number)
    return result

packets_file = generate_list_sum_to_n(2000,len(ben))

bdata = []

for (i,file) in enumerate(ben):
    d = np.load(file)
    ind = np.random.choice(range(0, len(d)-1), size=packets_file[i], replace=False)
    bdata.append(d[ind])
bdata = np.concatenate(bdata)

packets_file = generate_list_sum_to_n(6000,len(mal))
mdata = []

for (i,file) in enumerate(mal):
    d = np.load(file)
    ind = np.random.choice(range(0, len(d)-1), size=packets_file[i], replace=False)
    mdata.append(d[ind])
mdata = np.concatenate(mdata)
print(len(mdata))
print(len(bdata))
x = np.concatenate((bdata,mdata))
y = np.concatenate((np.zeros(len(bdata)),np.ones(len(mdata))))

shuffle = np.random.permutation(len(x))
x = x[shuffle]
y = y[shuffle]
print(y[:100])
print(x.shape,y.shape)
np.save("X_test.npy", x)
np.save("y_test.npy", y)


