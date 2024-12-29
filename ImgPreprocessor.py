from sys import stdout
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import hog, local_binary_pattern
from skimage.util import img_as_int
import skimage.transform as transform
from joblib import Parallel, delayed
from pathlib import Path
import logging
import time
import numpy as np

logger = logging.getLogger(__name__)

# Process single image
def processImage(filePath : Path, logFilePath : Path, currentId=0, totalCount=1):
    logging.basicConfig(filename=logFilePath, 
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

    hogFeature = hog(grayImage,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    visualize=False,
                    block_norm='L2-Hys'
                    )
    
    lbpRadius = 3
    lbpPoint = 3 * lbpRadius
    lbp = local_binary_pattern(img_as_int(grayImage), R=lbpRadius, P=lbpPoint)
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
def processAllImage(imgList, logFilePath : Path):
    xData = None
    yData = None
    processJobCount = 8

    logger.info(f"Processing all images with {processJobCount} jobs...")
    st = time.time()
    resultList = Parallel(n_jobs=processJobCount, return_as="generator_unordered")(
        delayed(processImage)(filePath, logFilePath, currentId, len(imgList)) 
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
    logger.info(f"Processing completed, costing {et - st:.2f} seconds")
    return xData, yData