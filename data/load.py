import numpy as np

def load_npy_data(n_packets = 1):
    X = np.load('data/kaggle/imgs0.npy')
    for i in range(1, n_packets):
        X = np.append(X, np.load('data/kaggle/imgs{}.npy'.format(i)), axis=0)
    Y_all = np.load('data/kaggle/labels.npy')

    Y = Y_all[0:(n_packets * 1000)][:]

    return X, Y