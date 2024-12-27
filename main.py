import skimage.feature as imgfeature
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
import skimage.transform as transform
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Get all image name
def getImageName(imgPathStr):
    print("Getting filenames...")
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
    index = 0
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
        print(f"processing image {index}...")
        index += 1
    return xData, yData

catPath = "../AnimalData/cat-db"
randomPath = "../AnimalData/random"

imgList = getImageName(catPath)
imgList.extend(getImageName(randomPath))
xData, yData = processImage(imgList)
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.2, random_state=114514)

scaler = StandardScaler()
xTrain = scaler.fit_transform(xTrain)
xTest = scaler.transform(xTest)

svm = SVC(kernel='rbf', random_state=114514)

params = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001]}
gridSearch = GridSearchCV(svm, params, cv=5, scoring='accuracy')
gridSearch.fit(xTrain, yTrain)

bestSvm = gridSearch.best_estimator_
bestSvm.fit(xTrain, yTrain)

yPred = bestSvm.predict(xTest)
accuracy = accuracy_score(yTest, yPred)
print(f"Accuracy: {accuracy:.2f}")
