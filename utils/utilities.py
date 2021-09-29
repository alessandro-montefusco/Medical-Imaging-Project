import os
import numpy as np
import cv2 as cv

####################################### INNER FUNCTIONS ######################################

def __find_information(image):
    '''
    The function computes the area and the perimeter of all the masses retrieved from the
    groundtruth directory of INbreast dataset.
    :param image: the current image to be computed.
    :return: a tuple of different elements:
        - list_area is a list with all the areas of the masses;
        - list_perimeter is a list with all the perimeters of the masses;
        - sum_area is the sum of all the areas;
        - sum_perimeter is the sum of all the perimeters.
    '''
    list_area = []
    list_perimeter = []
    sum_area = 0
    sum_perimeter = 0
    contours, _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        # Ignore isolated pixels or points that do not belong to the masses
        if (cv.contourArea(contours[i]) > 50 and cv.arcLength(contours[i], True) > 40):
            list_area.append(cv.contourArea(contours[i]))
            list_perimeter.append(cv.arcLength(contours[i], True))
            sum_area += cv.contourArea(contours[i])
            sum_perimeter += cv.arcLength(contours[i], True)

    return list_area, list_perimeter, sum_area, sum_perimeter

############################################################################################

def extract_information(ground_path):
    '''
    The function computes the minimum and maximum values of area and perimeter among all the
    masses and the average values of the area and perimeter.
    :param ground_path: the path of the directory in which we have the groundtruth images.
    :return: the parameters previously descripted.
    '''
    # A priori information about masses from the groundtruth of the whole dataset.
    ground_images = os.listdir(ground_path)
    list_areas = []
    list_perimeters = []
    sum_area_tot = 0
    sum_perimeter_tot = 0

    for ground in ground_images:
        img = cv.imread(ground_path + "\\" + ground, cv.IMREAD_GRAYSCALE)
        img = cv.resize(img, (512, 512))
        area, perimeter, sum_Area, sum_perimeter = __find_information(img)
        sum_area_tot += sum_Area
        sum_perimeter_tot += sum_perimeter

        list_areas.extend(area)
        list_perimeters.extend(perimeter)

    min_area = min(list_areas)
    min_perimeter = min(list_perimeters)

    max_area = max(list_areas)
    max_perimeter = max(list_perimeters)

    return min_area, max_area, min_perimeter, max_perimeter

def check_file():
    '''
    The function checks if the features and the labels have been computed and saved on their
    respectively files.
    :return: True if the files already exist, False otherwise.
    '''
    return os.path.exists("file\Features.txt") and os.path.exists("file\Labels.txt")

def load():
    '''
    The function loads the files with the features and labels.
    '''
    with open("file\Features.txt","r") as features_file:
        train_features = np.loadtxt(features_file)
    with open("file\Labels.txt","r") as labels_file:
        train_labels = np.loadtxt(labels_file)
    return train_features,train_labels

def store(train_features, train_labels):
    '''
    The function stores the features and the labels in two different files.
    :param train_features: the list of features extracted from all the images in the Training Set.
    :param train_labels: the list of all the labels associated to each image of the Training Set.
    '''
    with open("file\Features.txt", "w") as features_file:
        np.savetxt(features_file,train_features)
    with open("file\Labels.txt", "w") as labels_file:
        np.savetxt(labels_file,train_labels)
