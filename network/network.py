from tflearn import DNN
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from matplotlib import pyplot as plt
import numpy as np

network = input_data(shape=[None, 48, 48, 1], name='input')
network = conv_2d(network, nb_filter=32, filter_size=[7, 7], activation='relu')
network = max_pool_2d(network, kernel_size=2)
network = conv_2d(network, nb_filter=64, filter_size=[7, 7], activation='relu')
network = max_pool_2d(network, kernel_size=2)
network = fully_connected(network, n_units=512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, n_units=7, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='targets')
# abstract the network into a file
model = DNN(network)
model.load("model/model.tfl")


def predict(input_image):
    prediction = model.predict(input_image)
    print(prediction)

    predicted_class = np.argmax(prediction)
    print(predicted_class)

    meaning_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    print(meaning_list[predicted_class])

    plt.imshow(input_image, interpolation='bicubic')
    plt.title(meaning_list[predicted_class])
    plt.xticks([]), plt.yticks([])
    plt.show()
