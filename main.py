from sys import stdout
import skimage.feature as imgfeature
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import img_as_int
import skimage.transform as transform
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import random
from joblib import Parallel, delayed
import logging
import time

catPath = "../AnimalData/cat-db"
randomPath = "../AnimalData/random"
pictureCount = 1000
logger = logging.getLogger(__name__)

# Get all image name
def getImageName(imgPathStr):
    logger.info(f"Getting filenames from '{imgPathStr}'...")
    imgList = []
    imgPath = Path(imgPathStr)
    for item in imgPath.iterdir():
        if item.is_dir():
            continue
        else:
            imgList.append(item)
    return imgList

# Process single image
def processImage(filePath, currentId, totalCount):
    logging.basicConfig(stream=stdout, 
                    level=logging.INFO,
                    format='[%(asctime)s %(levelname)s] %(name)s:%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )
    logger.info(f"Processing image {filePath.name} ({currentId}/{totalCount})...")
    isCat = 0
    if filePath.name.startswith("cat"):
        isCat = 1

    try:
        image = imread(filePath)
    except OSError as e:
        logger.warning(f"Cannot open image file {filePath.name} bacause\n{e}")
        return None, None
    
    image = transform.resize(image, (128, 128))
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
    lbp = imgfeature.local_binary_pattern(img_as_int(grayImage), R=lbpRadius, P=lbpPoint)
    binsCount = int(lbp.max() + 1)
    lbpHist, _ = np.histogram(lbp.ravel(), 
                              density=True, 
                              bins=binsCount, 
                              range=(0, binsCount)
                              )
    
    totalFeature = np.concatenate((hogFeature, lbpHist))
    logger.info(f"File {filePath.name} processing completed")
    return totalFeature, isCat
    
# Process all image
def processAllImage(imgList):
    xData = None
    yData = None
    processJobCount = 8

    logger.info(f"Processing all images with {processJobCount} jobs...")
    st = time.time()
    resultList = Parallel(n_jobs=processJobCount, return_as="generator_unordered")(
        delayed(processImage)(filePath, currentId, len(imgList)) 
            for filePath, currentId in zip(imgList, range(1, len(imgList) + 1))
        )

    for totalFeature, isCat in resultList:
        if totalFeature is None or isCat is None:
            continue

        if xData is None:
            xData = totalFeature
        else:
            xData = np.vstack((xData, totalFeature))

        if yData is None:
            yData = [isCat]
        else:
            yData.append(isCat)
    et = time.time()
    logger.info(f"Processing completed, costing {et - st:.4f} seconds")
    return xData, yData

def main():
    logging.basicConfig(stream=stdout, 
                        level=logging.INFO,
                        format='[%(asctime)s %(levelname)s] %(name)s:%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S'
                        )
    logger.info("Program started")
    imgList = getImageName(catPath)
    imgList.extend(getImageName(randomPath))
    random.shuffle(imgList)
    imgList = imgList[0 : pictureCount]
    xData, yData = processAllImage(imgList)
    xTrain, xTest, yTrain, yTest = train_test_split(xData, 
                                                    yData, 
                                                    test_size=0.2, 
                                                    random_state=114514
                                                    )

    logger.info("Scaling X...")
    scaler = StandardScaler()
    xTrain = scaler.fit_transform(xTrain)
    xTest = scaler.transform(xTest)

    logger.info("Building SVM...")
    svm = SVC(kernel='rbf', C=100000, random_state=114514)

    logger.info("Fitting model...")
    svm.fit(xTrain, yTrain)

    logger.info("Predicting...")
    yPred = svm.predict(xTest)
    accuracy = accuracy_score(yTest, yPred)
    print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
