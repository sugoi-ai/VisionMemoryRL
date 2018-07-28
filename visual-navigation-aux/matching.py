import numpy as np
import cv2

from matplotlib import pyplot as plt


def match(obj, scene, min_match_count=10, flann_index_kdtree=0, min_dis_rat=0.7):
    surf = cv2.xfeatures2d_SIFT.create()
    kp1, des1 = surf.detectAndCompute(obj, None)
    kp2, des2 = surf.detectAndCompute(scene, None)

    index_params = dict(algorithm=flann_index_kdtree, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < min_dis_rat * n.distance:
            good.append(m)

    if len(good) > min_match_count:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        m, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h, w = obj.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, m)

        return [np.int32(dst)]
    else:
        return [0, 0, 0, 0]


if __name__ == "__main__":
    img1 = cv2.imread('img/scene1_obj2.png', 0)
    img2 = cv2.imread('img/ai2thor_scene1.jpg', 1)
    pts = match(img1, img2)
    if pts != [0, 0, 0, 0]:
        img2 = cv2.polylines(img2, pts, True, [0, 0, 255], 3, cv2.LINE_AA)

    rgb = img2[..., ::-1]
    plt.imshow(rgb), plt.show()
