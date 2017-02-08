"""

Reads an image from the hard drive and the classifies the emotion

"""

from tflearn import DNN
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


from os import listdir
from os.path import isfile, join


import numpy as np
import cv2
from matplotlib import pyplot as plt


# get image file and process it
image_path = 'data/images/img5.jpg'



def display_image(img):
    plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])
    plt.show()

def detect_face(raw_img):
    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

    minisize = (raw_img.shape[1], raw_img.shape[0])
    miniframe = cv2.resize(raw_img, minisize)

    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    print(faces)
    for f in faces:
        x, y, w, h = [v for v in f]
        img_facedetect = cv2.rectangle(raw_img, (x, y), (x + w, y + h), (255, 255, 255))

    return img_facedetect

def crop_face(raw_img):
    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

    minisize = (raw_img.shape[1], raw_img.shape[0])
    miniframe = cv2.resize(raw_img, minisize)

    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    print(faces)
    for f in faces:
        x, y, w, h = [v for v in f]
        img_facedeterct = cv2.rectangle(raw_img, (x, y), (x + w, y + h), (255, 255, 255))

        sub_face = raw_img[y:y + h, x:x + w]

    return sub_face

def resize(face, new_dim):
    r = new_dim / face.shape[1]
    dim = (new_dim, int(face.shape[0] * r))

    # perform the actual resizing of the image and show it
    resized = cv2.resize(face, dim, interpolation=cv2.INTER_AREA)
    return resized


def load_normalized(path, dim):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    img_array = np.empty([len(onlyfiles), dim, dim], dtype=int)
    for n in range(0, len(onlyfiles)):
        img_array[n][:][:] = cv2.imread(join(path, onlyfiles[n]), 0)
    return img_array





net = input_data(shape=[None, 48, 48, 1], name='input')
net = conv_2d(net, nb_filter=32, filter_size=[7, 7], activation='relu')
net = max_pool_2d(net, kernel_size=2)
net = conv_2d(net, nb_filter=64, filter_size=[7, 7], activation='relu')
net = max_pool_2d(net, kernel_size=2)
net = fully_connected(net, n_units=512, activation='relu')
net = dropout(net, 0.5)
net = fully_connected(net, n_units=7, activation='softmax')
net = regression(net, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='targets')
# abstract the network into a file
model = DNN(net)
model.load("model/model.tfl")




img_color = cv2.imread(image_path)
img = cv2.imread(image_path, 0)
display_image(img)
cropped_face = crop_face(img)
X = resize(cropped_face, 48)

X = X.reshape([1, 48, 48, 1])

prediction = model.predict(X)
print(prediction)


predicted_class = np.argmax(prediction)
print(predicted_class)

meaning_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
print(meaning_list[predicted_class])


plt.imshow(img_color, interpolation='bicubic')
plt.title(meaning_list[predicted_class])
plt.xticks([]), plt.yticks([])
plt.show()

"""
0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
"""
