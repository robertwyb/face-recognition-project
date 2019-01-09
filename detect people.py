import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import cv2
from PIL import Image
from sklearn.svm import SVC


def load_data():
    data, label = [], []
    num = 270
    path = "./train/"
    pid = ['r','c']
    for j in range(len(pid)):
        for number in range(num):
            path_full = path + pid[j] + str(number) +'.jpg'
            image = Image.open(path_full).convert('L')
            image = image.resize((100, 100), Image.ANTIALIAS)
            img = np.reshape(image, (1, 100*100))
            data.extend(img)
        label.extend(j * np.ones(num, dtype=np.int))
    data = np.reshape(data, (-1, 100*100))
    return np.matrix(data), np.matrix(label).T


def knn(neighbor, traindata, trainlabel, testdata):
    neigh = KNeighborsClassifier(n_neighbors=neighbor)
    neigh.fit(traindata, trainlabel)
    result = neigh.predict(testdata)
    return result


if __name__ == '__main__':

    # load data from train folder and pca feature extraction
    data, label = load_data()
    pca = PCA(n_components=0.9, whiten=True)
    pca_data = pca.fit_transform(data) 
    color = (255, 0, 0)
    cap = cv2.VideoCapture("test.mp4")
    while cap.isOpened():
        status, frame = cap.read()
        if not status:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        classifier = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

        # detect faces in the frame
        faces = classifier.detectMultiScale(frame_gray, 1.3, 5, minSize=(150, 150))
        if len(faces) > 0:
            for x, y, w, h in faces:
                # crop the face
                image = frame_gray[y - 10: y + h + 10, x - 10: x + w + 10]

                # resize the crop part then use knn to recognize
                image = cv2.resize(image, (100, 100))
                img_test = np.reshape(image, (1, 100 * 100))
                pca_test = pca.transform(img_test)
                result = knn(5, pca_data, label, pca_test)
                faceID = result[0]
                print(faceID)

                # find corresponding faces and print faces
                if faceID == 0:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)
                    cv2.putText(frame,'Robert', (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                elif faceID == 1:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)
                    cv2.putText(frame, 'Cheney', (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                else:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)
                    cv2.putText(frame, 'Unknown', (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("detect", frame)
        k = cv2.waitKey(25)
        if k & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
