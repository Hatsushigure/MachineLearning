import skimage.feature as imgfeature
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
import skimage.transform as transform
import matplotlib.pyplot as plt
from pathlib import Path
import random

# Get all image name
def getImageName(imgPathStr):
    imgList = []
    imgPath = Path(imgPathStr)
    for item in imgPath.iterdir():
        if item.is_dir():
            continue
        else:
            imgList.append(item)
    return imgList

# Process image
def processImage(imgList):
    xData = None
    yData = None
    for filename in imgList:
        isCat = 0
        if filename.name.startswith("cat"):
            isCat = 1

        image = imread(filename)
        image = transform.resize(image, (256, 256))
        grayImage = rgb2gray(image)

        hogFeature = imgfeature.hog(grayImage,
                         orientations=9,
                         pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2),
                         visualize=False,
                         block_norm='L2-Hys'
                         )

        lbpRadius = 3
        lbpPoint = 3 * lbpRadius
        lbp = imgfeature.local_binary_pattern(grayImage, R=lbpRadius, P=lbpPoint)
        binsCount = int(lbp.max() + 1)
        lbpHist, _ = np.histogram(lbp.ravel(), density=True, bins=binsCount, range=(0, binsCount))

        totalFeature = np.concatenate((hogFeature, lbpHist))

        if xData is None:
            xData = totalFeature
        else:
            xData = np.vstack((xData, totalFeature))

        if yData is None:
            yData = np.vstack([isCat])
        else:
            yData = np.vstack((yData, [isCat]))
    return xData, yData

imgPathStr = "../AnimalData/downloaded"
trainPercentage = 0.8

imgList = getImageName(imgPathStr)
random.shuffle(imgList)
trainImgList = imgList[0 : int(trainPercentage * len(imgList))]

xTrain, yTrain = processImage(trainImgList)
