import cv2
import glob
import os
import numpy as np
from skimage.feature import hog
from sklearn.cluster import KMeans, AgglomerativeClustering as AC, DBSCAN
from sklearn import metrics


path = 'Folio\*\*.jpg'
folder = 'Mixed Folio'
if not os.path.exists(folder):
    os.makedirs(folder)
folder_path = folder + '\*\*.jpg'
leafname = []
dataset = []
label = []
feature = []


def convert3D(bin_image):
    _3D = []
    for row in bin_image:
        new = [[x] * 3 for x in row]
        _3D.append(new)
    return np.array(_3D)

def extractROI(image):
    # resize
    max_dimension = max(image.shape)
    scale = 700 / max_dimension
    image = cv2.resize(image, None, fx=scale, fy=scale)

    # smooth image
    image_blur = cv2.GaussianBlur(image, (7, 7), 0)

    # grayscale
    gray = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)

    # binary
    if gray[gray.shape[0] // 2][gray.shape[1] // 2] > 95:
        _, bin_img = cv2.threshold(gray, 190, 1, cv2.THRESH_BINARY_INV)
    else:
        _, bin_img = cv2.threshold(gray, 130, 1, cv2.THRESH_BINARY_INV)

    # dilation
    kernel = np.ones((4, 1), np.uint8)
    img_dilation = cv2.dilate(bin_img, kernel, iterations=1)
    img_dilation = convert3D(img_dilation)

    # roi
    roi = image*img_dilation
    return roi

def preprocess(path):
    max_width = 0

    # resize + findmax width
    for image_path in glob.glob(path):
        head, name = os.path.split(image_path)
        _, subfolder = os.path.split(head)
        dir = os.path.join(folder, subfolder)
        if not os.path.exists(dir):
            os.makedirs(dir)

        image = cv2.imread(image_path)
        image = extractROI(image.copy())
        #max_dimension = max(image.shape)
        #scale = 700 / max_dimension
        #image = cv2.resize(image, None, fx=scale, fy=scale)

        cv2.imwrite(os.path.join(dir, name), image)

        # find max width
        if image.shape[1] > max_width:
            max_width = image.shape[1]

    # padding
    for file in glob.glob(folder_path):
        img = cv2.imread(file)
        margin = (max_width - img.shape[1])//2
        pad = np.zeros([700, max_width, 3])
        pad[:, margin:margin+img.shape[1]] = img
        cv2.imwrite(file, pad)

def loadname(path):
    file = open(path)
    for name in file:
        leafname.append(name.strip())

def load_data(folder_path):
    for file in glob.glob(folder_path):
        img = cv2.imread(file)
        dataset.append(img)

        # label:
        head, _ = os.path.split(file)
        _, subfolder = os.path.split(head)
        label.append(leafname.index(subfolder))

def featureExtr():
    for img in dataset:
        # color - hsv
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        c = sum(sum(hsv[:, :, 0]))/(hsv.shape[0]*hsv.shape[1])

        # shape - hog
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, _ = hog(gray, orientations=8, pixels_per_cell=(16, 16),
            cells_per_block=(1, 1), visualize=True, multichannel=False)

        np.insert(h, 0, c)
        feature.append(h)

def folder_result(f, pred_label):
    if not os.path.exists(f):
        os.makedirs(f)
    for i in range(len(pred_label)):
        sub = os.path.join(f, str(pred_label[i]))
        if not os.path.exists(sub):
            os.makedirs(sub)
        cv2.imwrite(os.path.join(sub, str(i)+'.jpg'), dataset[i])

def Kmeans():
    kmeans = KMeans(n_clusters=32, random_state=0).fit(feature)
    pred_label = kmeans.labels_
    print('silhouette_score = ', metrics.silhouette_score(feature, pred_label, metric='euclidean'))
    print('homogeneity_score = ', metrics.homogeneity_score(label, pred_label.tolist()))
    #print(kmeans.cluster_centers_)
    #print(kmeans.labels_)

    folder_result('Kmeans', pred_label)

def HAC():
    hac = AC(n_clusters=32).fit(feature)
    pred_label = hac.labels_
    print('silhouette_score = ', metrics.silhouette_score(feature, pred_label, metric='euclidean'))
    print('homogeneity_score = ', metrics.homogeneity_score(label, pred_label.tolist()))
    #print(hac.labels_)

    folder_result('HAC', pred_label)

def DBS():
    dbscan = DBSCAN(eps=12, min_samples=2).fit(feature)
    pred_label = dbscan.labels_
    print('silhouette_score = ', metrics.silhouette_score(feature, pred_label, metric='euclidean'))
    print('homogeneity_score = ', metrics.homogeneity_score(label, pred_label.tolist()))
    #print(dbscan.labels_)

    folder_result('DBSCAN', pred_label)


#preprocess(path)
loadname('leaf_name')
load_data(folder_path)
featureExtr()
feature = np.array(feature)
Kmeans()
HAC()
DBS()