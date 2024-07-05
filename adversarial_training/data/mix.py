import numpy as np
np.random.seed(42)
# Assuming x_attack and z_packets_2 are 2D arrays and y_attack is a 1D array
x_attack = np.load("test.npy")
y_attack = np.ones(len(x_attack))
z_packets_2 = np.load("/home/sura/working_directory/modified_bars/data/train/complete_weekday/X_train.npy")

nop = 10 #number of 1 labeled packets


x_ind = np.random.choice(range(0, len(x_attack)-1), size=nop, replace=False)
x_attack = x_attack[x_ind]
y_attack = y_attack[x_ind]



# nop = 100#number of 0 labeled packets


z_ind = np.random.choice(range(0, len(z_packets_2)-1), size=nop, replace=False)
z_packets_2 = z_packets_2[z_ind]
z_y = np.zeros(len(z_packets_2))

x = np.concatenate((x_attack,z_packets_2))
y = np.concatenate((y_attack,z_y))

shuffle = np.random.permutation(len(x))
x = x[shuffle]
y = y[shuffle]

print(x.shape,y.shape)
np.save("X_test_mix.npy", x)
np.save("y_test_mix.npy", y)
