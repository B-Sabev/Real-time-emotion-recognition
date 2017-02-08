from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
import numpy as np
import cv2


def display_image(image):
    plt.imshow(image, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])
    plt.show()


def detect_face(image):
    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

    minisize = (image.shape[1], image.shape[0])
    miniframe = cv2.resize(image, minisize)

    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    print(faces)
    for face in faces:
        """Face rectangle(width*height) starts at (x,y)"""
        x_coordinate, y_coordinate, face_width, face_height = [v for v in face]
        image_facedetect = cv2.rectangle(image,
                                         (x_coordinate, y_coordinate),
                                         (x_coordinate + face_width, y_coordinate + face_height),
                                         (255, 255, 255))

    return image_facedetect


def crop_face(image):
    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

    minisize = (image.shape[1], image.shape[0])
    miniframe = cv2.resize(image, minisize)

    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    print(faces)
    for face in faces:
        x_coordinate, y_coordinate, face_width, face_height = [v for v in face]
        # image_facedetect = cv2.rectangle(image, (x_coordinate, y_coordinate), (x_coordinate + face_width, y_coordinate + face_height), (255, 255, 255))

        sub_face = image[y_coordinate:y_coordinate + face_height, x_coordinate:x_coordinate + face_width]

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
