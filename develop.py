import cv2
import os
folder = 'dataset/mask'
names = sorted(os.listdir(folder))
for name in names:
    # path = folder + '/' + name
    # oppure
    path = os.path.join(folder, name)
    im = cv2.imread(path,1) # 1
    im = cv2.resize(im, (64,64))
    cv2.imshow("im", im)
    cv2.waitKey(33)