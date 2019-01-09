import cv2


def capture_trainset(video_name, window_name, pic_num, path_name):
    cv2.namedWindow(window_name)
    cap = cv2.VideoCapture(video_name)
    classfier = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    color = (255, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    num = 0
    while cap.isOpened():
        status, frame = cap.read() 
        if not status:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # use classifier to detect faces in the frame
        faces = classfier.detectMultiScale(gray, 1.3, 2, minSize=(300, 300))
        if len(faces) != 0:
            for x, y, w, h in faces:
                img_name = "%s/%s%d.jpg" % (path_name, video_name[0], num)
                print(img_name)
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                cv2.imwrite(img_name, image)
                num += 1

                # when capture enough pictures, end the program
                if num > pic_num:
                    break

                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                cv2.putText(frame, 'saved:%d/%d' % (num, pic_num), (x + 10, y + 10), font, 1, color, 4)
        if num > pic_num:
            break

        cv2.imshow(window_name, frame)
        c = cv2.waitKey(25)
        if c & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    capture_trainset("r.mp4", "get face", 499, "./train")
