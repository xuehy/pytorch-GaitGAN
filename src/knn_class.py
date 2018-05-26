from sklearn.neighbors import KNeighborsClassifier
import cv2
import os
import numpy as np
angles = ['090', '000', '018', '036', '054', '072',
          '108', '126', '144', '162', '180']
pid = 63
X = []
y = []
for cond in ['nm-01', 'nm-02', 'nm-03', 'nm-04']:
    for p in range(pid, 125):
        for ang in angles:
            path = '../data/GEI_CASIA_B/gei/%03d/%s/%03d-%s-%s.png' % (
                p, cond, p, cond, ang)
            path1 = './transformed_28500/%03d-%s-%s.png' % (p, cond, ang)
            if not os.path.exists(path):
                continue
            if ang == '090':
                img = cv2.imread(path, 0)

            else:
                img = cv2.imread(path1, 0)
            img = cv2.resize(img, (64, 64))
            img = img.flatten().astype(np.float32)
            X.append(img)
            y.append(p-63)

nbrs = KNeighborsClassifier(n_neighbors=1, p=1, weights='distance')
X = np.asarray(X)
y = np.asarray(y).astype(np.int32)
nbrs.fit(X, y)

testX = []
testy = []
pid = 63
for cond in ['nm-05', 'nm-06']:
    for p in range(pid, 125):
        for ang in angles:
            path = '../data/GEI_CASIA_B/gei/%03d/%s/%03d-%s-%s.png' % (
                p, cond, p, cond, ang)
            path1 = './transformed_28500/%03d-%s-%s.png' % (p, cond, ang)
            if not os.path.exists(path):
                continue
            if ang == '090':
                img = cv2.imread(path, 0)
            else:
                img = cv2.imread(path1, 0)
            img = cv2.resize(img, (64, 64))
            img = img.flatten().astype(np.float32)
            testX.append(img)
            testy.append(p-63)

testX = np.asarray(testX).astype(np.float32)
print(nbrs.score(testX, testy))
