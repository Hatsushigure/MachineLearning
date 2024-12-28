from sys import stdout
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from ImgPreprocessor import processAllImage
from joblib import dump
import random
import logging
import argparse

catPath = "../AnimalData/cat-db"
randomPath = "../AnimalData/random"
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

def getImgData(count):
    imgList = getImageName(catPath)
    imgList.extend(getImageName(randomPath))
    random.shuffle(imgList)
    imgList = imgList[0 : count]
    return processAllImage(imgList)

def main():
    logging.basicConfig(stream=stdout, 
                        level=logging.INFO,
                        format='[%(asctime)s %(levelname)s] %(name)s:%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S'
                        )
    logger.info("Program started")

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--datasetSize", "-s", required=True, type=int)
    argparser.add_argument("--dumpModel", "-d", default=True, type=bool)
    argNamespace = argparser.parse_args()
    pictureCount = argNamespace.datasetSize
    doDumpModel = argNamespace.dumpModel
 
    xData, yData = getImgData(pictureCount)
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

    if doDumpModel:
        logger.info(f"Model saved to ./generated/svm.{pictureCount}.dmp")
        if not Path("./generated").exists():
            Path.mkdir("./generated")
        dump(svm, f"./generated/svm.{pictureCount}.dmp")

if __name__ == "__main__":
    main()
