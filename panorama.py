import cv2
import matplotlib.pyplot as plt
import numpy as np


# helper function to plt the image
def plt_show(image):
    image = cv2.cv2tColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image, cmap = 'gray')
    plt.show()


"""
function to create panorama image 
"""
def stitch():
    img1 = cv2.imread("landscape_1.jpg")
    img2 = cv2.imread("landscape_2.jpg")
    img3 = cv2.imread("landscape_3.jpg")
    img4 = cv2.imread("landscape_4.jpg")
    img5 = cv2.imread("landscape_5.jpg")
    img6 = cv2.imread("landscape_6.jpg")
    img7 = cv2.imread("landscape_7.jpg")
    img8 = cv2.imread("landscape_8.jpg")
    img9 = cv2.imread("landscape_9.jpg")
    out1 = img5
    for i in [img6, img7, img8, img9]:
        out1 = basic_wrap(out1, i)
    plt_show(out1)
    cv2.imwrite('rightside.jpg', out1)
    list2 = [np.flip(img4, axis=1), np.flip(img3, axis=1), np.flip(img2, axis=1), np.flip(img1, axis=1)]
    out2 = np.flip(out1, axis=1)
    for i in range(4):
        out2 = basic_wrap(out2, list2[i])
    out2 = np.flip(out2, axis=1)
    plt_show(out2)
    cv2.imwrite("result.jpg", out2)


def basic_wrap(B, A):

    sift = cv2.xfeatures2d.SIFT_create()
    key1, desc1 = sift.detectAndCompute(A, None)
    key2, desc2 = sift.detectAndCompute(B, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    src = np.array([key1[m.queryIdx].pt for m in good])
    dest = np.array([key2[m.trainIdx].pt for m in good])

    src = (src.reshape(-1, 1, 2)).astype(np.float)
    dest = (dest.reshape(-1, 1, 2)).astype(np.float)

    M, mask = cv2.findHomography(src, dest, cv2.RANSAC, 5.0)
    img_out = cv2.warpPerspective(A, M, (B.shape[1] + 700, B.shape[0] + 20))
    img_out[0: B.shape[0], 0: B.shape[1]] = B
    # plt_show(img_out)
    trim = np.where(~img_out.any(axis=0))[0]
    # print(np.min(trim), np.max(trim))
    for i in range(len(trim)):
        try:
            img_out = np.delete(img_out, np.min(trim) - 10, axis=1)
        except:
            break
    # plt_show(img_out)
    origin = cv2.warpPerspective(A, M, (B.shape[1] + 700, B.shape[0] + 20))
    # plt_show(origin)
    img_paste = origin[0: B.shape[0], B.shape[1] - 100: B.shape[1] + 100]
    img_paste_crop = img_paste[10:, :]
    # plt_show(img_paste_crop)
    print(img_paste.shape, img_paste_crop.shape)
    mask = 255 * np.ones(img_paste_crop.shape, img_paste_crop.dtype)
    h, w = int(B.shape[1]), int(B.shape[0]/2)
    center = (h, w+5)
    img_new = cv2.seamlessClone(img_paste_crop, img_out, mask=mask, p=center, flags=cv2.NORMAL_CLONE, blend=None)
    # plt_show(img_new)
    cv2.imwrite('temp.jpg', img_new)
    return img_new




stitch()

