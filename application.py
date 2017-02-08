"""

Reads an image from the hard drive and the classifies the emotion

"""

import image.image as img
import network.network as net
import cv2

# get image file and process it
image_path = 'data/images/img5.jpg'

image_color = cv2.imread(image_path, 1)
image_grayscale = cv2.imread(image_path, 0)
img.display_image(image_grayscale)
cropped_face = img.crop_face(image_grayscale)
resized_face = img.resize(cropped_face, 48)
input_image = resized_face.reshape([1, 48, 48, 1])

net.predict(input_image=input_image)
