import numpy as np
import cv2
import matplotlib.pyplot as plt
# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


def plt_show(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image, cmap = 'gray')
    plt.show()


"""
parameter of camera:
1/2.3-inch (6.17 x 4.55 mm)
Wide FOV 16.8mm
Medium FOV	21.4mm
Narrow FOV	32.3mm

formula used to compute distance: 
Distance = focal length * real height * image height / (object height * sensor height)
sensor size 22.5 x 15 mm
image size 1000 x 667 pixel
"""
f = 16
real_height = 62
image_height = 1000
sensor_height = 22.5


def compute_distance(d):
    return f * real_height * image_height / (d * sensor_height)


def get_eye_distance(img):
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 4, minSize=(100, 100))
    print(faces)
    color = (255, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_gray = gray[y:y + h, x:x + w]
        face_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(face_gray)
        maybe_eyes = []
        for ex, ey, ew, eh in eyes:
            if ey < h/2:
                maybe_eyes.append((ex, ey, ew, eh))
        print(maybe_eyes)
        real_eyes = []
        for i in range(len(maybe_eyes)):
            for j in range(len(maybe_eyes)):
                if i != j:
                    if abs(maybe_eyes[i][0] - maybe_eyes[j][0]) > 0.01 * w and abs(maybe_eyes[i][1] - maybe_eyes[1][0]) < 0.1 * h:
                        real_eyes = [(maybe_eyes[i], maybe_eyes[j])]
        if len(real_eyes) < 2:
            real_eyes = maybe_eyes[:2]
        eye_dist = abs(real_eyes[0][0] - real_eyes[1][0])
        dist = compute_distance(eye_dist)
        for ex, ey, ew, eh in real_eyes:
            cv2.rectangle(face_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        cv2.putText(img, 'distance: %dcm' % (dist/10), (x + 10, y - 20), font, 1, color, 3)
    plt_show(img)
    return None


get_eye_distance('test.jpg')


