"""

Starts the webcam and classifies emotion in real time


"""






import numpy as np
import cv2
from matplotlib import pyplot as plt
from time import sleep
from tflearn import DNN
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

def crop_face(raw_img):
    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

    minisize = (raw_img.shape[1], raw_img.shape[0])
    miniframe = cv2.resize(raw_img, minisize)

    faces = face_cascade.detectMultiScale(raw_img, 1.3, 5)
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



cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

meaning_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = gray[y:y + h, x:x + w]
        ret, frame = cap.read()
        sleep(1)
        if ret == True:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # what if the image is lost here
            face = crop_face(gray_frame)
            face = resize(face, 48)
            print(face.shape)
            face = face.reshape([1, 48, 48, 1])
            prediction = model.predict(face)
            predicted_class = np.argmax(prediction)
            print(meaning_list[predicted_class])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



def display_image(img):
    plt.imshow(img, cmap = 'gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])
    plt.show()

display_image(face)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()