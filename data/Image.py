"""

Object for images, NEEDS TO BE FINISHED, DOESN"T DO ANYTHING AND IT ISN'T USED ANYWERE


"""




from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
from matplotlib import pyplot as plt


class Image():
    """
    raw_image
    face
    normalized
    """
    def __init__(self, imagepath, ):
        self.raw_image
        self.face
        self.normalized






    def display_image(self):
        plt.imshow(self, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

img = cv2.imread('IMG_1.jpg',0)




def detect_face(raw_img):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    minisize = (raw_img.shape[1], raw_img.shape[0])
    miniframe = cv2.resize(raw_img, minisize)

    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    print(faces)
    for f in faces:
        x, y, w, h = [v for v in f]
        img_facedetect = cv2.rectangle(raw_img, (x, y), (x + w, y + h), (255, 255, 255))

    return img_facedetect

def crop_face(raw_img):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

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

face = crop_face(img)

print(face.shape)

resized = resize(face, 48)

display_image(resized)

print(resized.shape)
print("{}".format(resized))

# save face
cv2.imwrite("thumbnail.png", resized)

"""
mypath='images'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
    images[n] = cv2.imread( join(mypath,onlyfiles[n]), 0)


# TODO abstracting into objects
# TODO handling of multiple or no faces in the pictures, how to get the same data from a live stream
i = 0
for img in images:
    display_image(detect_face(img))
    normalized = resize(crop_face(img), 48)
    display_image(normalized)
    cv2.imwrite("img{}.png".format(i), normalized)
    i += 1
"""


def load_normalized(path, dim):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    img_array = np.empty([len(onlyfiles), dim, dim], dtype=int)
    for n in range(0, len(onlyfiles)):
        img_array[n][:][:] = cv2.imread(join(path, onlyfiles[n]), 0)
    return img_array


img_array = load_normalized('images/norm')

print(img_array.shape)