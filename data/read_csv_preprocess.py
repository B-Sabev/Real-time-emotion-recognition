"""

Reads the csv file from the Kaggle Facial Expressions dataset, cleans it and transforms it to a npy file


"""

import pandas as pd
import numpy as np




filepath = 'data/fe_Kaggle/fer2013/fer2013.csv'

def to_one_hot(vector):
    num_classes = np.unique(vector).shape[0]
    num_labels = vector.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    one_hot_vector = np.zeros((num_labels, num_classes))
    one_hot_vector.flat[index_offset + vector.ravel()] = 1
    return one_hot_vector

# Training set is 28,709
DATA_SIZE = 28000
data = pd.read_csv(filepath, nrows=DATA_SIZE)

train_df = data[data['Usage'] == "Training"].drop('Usage', axis=1)

imgs = train_df['pixels']
imgs.dropna()
labels = train_df['emotion'].values

X = np.zeros([DATA_SIZE, 48*48], float)
print(X.shape)
print(X.shape[0])
Y = np.asarray(labels)
Y = to_one_hot(Y)

for i in range(X.shape[0]):
    list = imgs[i].split()
    X[i][:] = np.asarray(list)

print("{}".format(X))

i = 0
for x in np.split(X, 28, axis=0):
    np.save('data/kaggle/imgs{}.npy'.format(i), x)
    i += 1

np.save('data/kaggle/labels.npy', Y)


