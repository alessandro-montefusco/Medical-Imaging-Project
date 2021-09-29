import cv2 as cv
import numpy as np
from utils.data_preprocessing import build_true_path
import math

############################INNER FUNCTION##################################

def __jaccard_similarity(im_true, im_pred):
    '''
    The function computes the Jaccard index.
    :param im_true: the true groundtruth of the current image.
    :param im_pred: the predicted groundtruth of the current image.
    :return: the Jaccard index.
    '''
    if im_true.shape != im_pred.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    intersection = np.logical_and(im_true, im_pred)

    union = np.logical_or(im_true, im_pred)
    return intersection.sum() / union.sum()

#############################################################################

    
def jaccard(pred_ground_images, path_images, path_gound_test):
    '''
    The function computes the Jaccard indexes for all the predicted images.
    :param path_ground_images: the list of groundtruths of the predicted masses.
    :param path_images: the list of paths of the images.
    :return: the list with all the Jaccard indexes;
    '''
    print("-------------------- [STATUS] Computing Jaccard index ----------------")
    jaccard_list = []


    for i in range(len(pred_ground_images)):
        _type = path_images[i].split("_").pop(0)
        path = build_true_path(path_images[i])

        ground_test_img = cv.imread(path_gound_test + "\\" + path, cv.IMREAD_GRAYSCALE)
        ground_test_img = cv.resize(ground_test_img,(512, 512))

        ground_test_img = np.asarray(ground_test_img).astype(np.bool)
        img_pred = cv.resize(pred_ground_images[i], (512, 512))
        img_pred = np.asarray(img_pred).astype(np.bool)
        
        jaccard = __jaccard_similarity(ground_test_img, img_pred)

        #check if the index is NaN (possible situation between a false positive and a groundtruth of a no-mass image)
        if math.isnan(jaccard) and _type == "NOMASS":
            jaccard = 1
        if math.isnan(jaccard) and _type == "MASS":
            jaccard = 0
        jaccard_list.append(jaccard)

    return jaccard_list