from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from pathlib import Path
from joblib import load
import logging
import time

logger = logging.getLogger(__name__)

def loadModel(filePath : Path) -> SVC:
    logger.info(f"Loading model from '{filePath}'...")
    if not filePath.exists:
        logger.error(f"Model file '{filePath}' does not exist!")
        return None
    model = load(filePath)
    return model

def trainModel(xData, yData):
    xTrain, xTest, yTrain, yTest = train_test_split(xData, 
                                                    yData, 
                                                    test_size=0.2, 
                                                    random_state=114514
                                                    )
    
    logger.info("Scaling X...")
    scaler = StandardScaler()
    # xTrain = scaler.fit_transform(xTrain)
    # xTest = scaler.transform(xTest)

    logger.info("Building SVM...")
    svm = SVC(kernel='rbf', C=100000, random_state=114514)

    logger.info("Fitting model...")
    st = time.time()
    svm.fit(xTrain, yTrain)
    et = time.time()
    logger.info(f"Fitting completed, costing {et - st:.2f} seconds")

    logger.info("Counting scores...")
    yPred = svm.predict(xTest)
    accuracy = accuracy_score(yTest, yPred)
    logger.info(f"Accuracy: {accuracy:.2f}")

    return svm
