import numpy as np

np.random.seed(31)

ben = ["weekday.npy",]
mal = ["adv/cam_ack.npy","adv/cam_syn.npy","adv/cam_udp.npy"]


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

packets_file = generate_list_sum_to_n(33,len(ben))
bdata = []

for (i,file) in enumerate(ben):
    d = np.load(file)
    ind = np.random.choice(range(0, len(d)), size=packets_file[i], replace=False)
    ind.sort()
    bdata.append(d[ind])
bdata = np.concatenate(bdata)

packets_file = generate_list_sum_to_n(67,len(mal))
mdata = []

for (i,file) in enumerate(mal):
    d = np.load(file)
    ind = np.random.choice(range(0, len(d)-1), size=packets_file[i], replace=False)
    ind.sort()
    mdata.append(d[ind])
mdata = np.concatenate(mdata)

x = np.concatenate((bdata,mdata))
y = np.concatenate((np.zeros(len(bdata)),np.ones(len(mdata))))

#shuffle = np.random.permutation(len(x))
#x = x[shuffle]
#y = y[shuffle]

print(x.shape,y.shape)
np.save("X_4_adv.npy", x)
np.save("y_4_adv.npy", y)


