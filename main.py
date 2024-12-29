from pathlib import Path
from ImgPreprocessor import processAllImage, processImage
from joblib import dump
from ModelGenerator import trainModel, loadModel
import random
import logging
import argparse
import datetime
import numpy as np

catPath = "../AnimalData/cat-db"
randomPath = "../AnimalData/random"
logger = logging.getLogger(__name__)
currentDatetime = datetime.datetime.now()
logFilePath = Path(f"./logs/{currentDatetime.year}-{currentDatetime.month:0>2}-{currentDatetime.day:0>2}_{currentDatetime.hour:0>2}_{currentDatetime.minute:0>2}_{currentDatetime.second:0>2}.log")

# Get all image name
def getImageName(imgPathStr : str):
    logger.info(f"Getting filenames from '{imgPathStr}'...")
    imgList = []
    imgPath = Path(imgPathStr)
    for item in imgPath.iterdir():
        if item.is_dir():
            continue
        else:
            imgList.append(item)
    return imgList

# Get data for training model
def getTrainImgData(count : int):
    imgList = getImageName(catPath)
    imgList.extend(getImageName(randomPath))
    random.shuffle(imgList)
    imgList = imgList[0 : count]
    return processAllImage(imgList, logFilePath)

# Runs in train mode
def trainMode(pictureCount, doDumpModel):
    logger.info(f"Program is working in training mode, with dataset size {pictureCount}, and will {"" if doDumpModel else "not "}save the model")

    xData, yData = getTrainImgData(pictureCount)
    svm = trainModel(xData, yData)

    if doDumpModel:
        logger.info(f"Model saved to ./generated/svm.{pictureCount}.dmp")
        if not Path("./generated").exists():
            Path.mkdir("./generated")
        dump(svm, f"./generated/svm.{pictureCount}.dmp")

def main():
    logging.basicConfig(filename=logFilePath,
                        level=logging.INFO,
                        format='[%(asctime)s %(levelname)s] %(name)s:%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S'
                        )

    argparser = argparse.ArgumentParser(prog="CatRecognition",
                                        description="A program to train models and recognize image with cats"
                                        )
    subparsers = argparser.add_subparsers(title="subcommands",
                                          help="Available working mode",
                                          dest="workingMode"
                                          )
    subparserTrain = subparsers.add_parser("train", 
                                           help="Training models based on data"
                                           )
    subparserTrain.add_argument("--datasetSize", 
                           "-s", 
                           required=True, 
                           type=int,
                           help="Count of images chosen to train the model (required)"
                           )
    subparserTrain.add_argument("--dumpModel", 
                           "-d", 
                           choices=["on", "off"], 
                           default="on",
                           help="Whether to dump the model to file"
                           )
    subparserPredict = subparsers.add_parser("predict",
                                             help="Predict image content with trained model"
                                             )
    subparserPredict.add_argument("--model", 
                                  "-m",
                                  metavar="modelPath",
                                  required=True,
                                  help="The path to model file"
                                  )
    subparserPredict.add_argument("imagePath",
                                  help="Path to the image file to predict")
    argNamespace = argparser.parse_args()
    workingMode = argNamespace.workingMode
 
    logger.info("Program started")

    if workingMode == "train":
        pictureCount = argNamespace.datasetSize
        doDumpModel = True if argNamespace.dumpModel == "on" else False
        trainMode(pictureCount, doDumpModel) 
    elif workingMode == "predict":
        logger.info(f"Program is working in training mode")
        modelPath = Path(argNamespace.model)
        imagePath = Path(argNamespace.imagePath)
        logger.info(f"Image path: '{imagePath}'")
        logger.info(f"Model path: '{modelPath}'")
        processedImage, _ = processImage(imagePath, logFilePath)
        processedImage = np.vstack((processedImage,))
        svm = loadModel(Path(modelPath))
        result = svm.predict(processedImage)
        print(f"Result: {result[0]}")

if __name__ == "__main__":
    main()
