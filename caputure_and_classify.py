"""

Starts the webcam and classifies emotion in real time


"""
import numpy as np
import cv2
from time import sleep
import image.image as img
import network.network as net

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

meaning_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

while (True):
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
            face = img.crop_face(gray_frame)
            face = img.resize(face, 48)
            print(face.shape)
            face = face.reshape([1, 48, 48, 1])
            prediction = net.predict(face)
            predicted_class = np.argmax(prediction)
            print(meaning_list[predicted_class])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Display the resulting frame
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

img.display_image(face)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
