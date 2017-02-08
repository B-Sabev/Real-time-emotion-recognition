"""

Traing the network on the kaggle dataset

"""





import numpy as np
from sklearn.model_selection import train_test_split
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from data.load import load_npy_data

img, labels = load_npy_data(n_packets=1) # Max 28

print(img.shape)
print(labels.shape)



img = img.reshape([-1, 48, 48, 1])

net = input_data(shape=[None, 48, 48, 1], name='input')
net = conv_2d(net, nb_filter=32, filter_size=[7, 7], activation='relu')
net = max_pool_2d(net, kernel_size=2)
net = conv_2d(net, nb_filter=64, filter_size=[7, 7], activation='relu')
net = max_pool_2d(net, kernel_size=2)
net = fully_connected(net, n_units=512, activation='relu')
net = dropout(net, 0.5)
net = fully_connected(net, n_units=7, activation='softmax')
net = regression(net, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(net,
                    tensorboard_verbose=3,
                    tensorboard_dir="logs",
                    checkpoint_path='model/model.tfl.ckpt',
                    max_checkpoints=2,)

model.fit({'input': img}, {'targets': labels},
          n_epoch=1,
          validation_set=0.2,
          snapshot_step=500,
          show_metric=True,
          run_id='mnist')


model.save("model/model.tfl")