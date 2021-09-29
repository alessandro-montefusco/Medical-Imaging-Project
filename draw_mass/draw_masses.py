import cv2 as cv
import numpy as np
import random as rng
#from utils.data_preprocessing import build_true_path
from utils.utilities import extract_information

###################### GLOBAL INFORMATION ######################
ground_path = "dataset\groundtruth"
min_area, average_area, max_area, min_perimeter, average_perimeter, max_perimeter = extract_information(ground_path)
################################################################

###################################### INNER FUNCTIONS ######################################
def __check_masses(contours):
    new_contours = []
    if len(contours) < 1:
        return new_contours
    else:
        for i in range(len(contours)):
            check_perimeter = cv.arcLength(contours[i], True) > min_perimeter and cv.arcLength(contours[i], True) < max_perimeter
            check_area = cv.contourArea(contours[i]) > min_area and cv.contourArea(contours[i]) < max_area
            if check_area and check_perimeter:
                new_contours.append(contours[i])
        return new_contours

def __set_threshold(threshold_image, thr_value):
    # Threshold: 122 is an empirical value
    ret, thr_img = cv.threshold(threshold_image, thr_value, 255, 0, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # Opening
    kernel = np.ones((5, 5), np.uint8)
    thr_img = cv.morphologyEx(thr_img, cv.MORPH_ELLIPSE, kernel)
    # Contours
    _, contours, _ = cv.findContours(thr_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours
#############################################################################################

def my_draw_contours(segmeted_images):
    print("-------------------- [STATUS] Drawing contours -----------------------")
    ground_images = []
    outcomes = []
    for img in segmeted_images:
        contours = __set_threshold(img, 122)
        # Checks value below 122. 105 is another empirical value.
        contours = __check_masses(contours)

        if len(contours) == 0:
            contours = __set_threshold(img, 105)
            contours = __check_masses(contours)

        drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        for i in range(len(contours)):
            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
            # Draw groundtruth
            cv.drawContours(img, contours, i, color, 2)
            # Draw masses on img
            cv.drawContours(drawing, contours, i, (255, 255, 255), -1)  # -1 = FILLED
        outcomes.append(img)
        ground_images.append(drawing)
    print("-------------------- [NOTIFY] All images have been processed ---------")
    return  outcomes, ground_images

def clean_unet_images(input_unet_images, output_unet_images):
    print("-------------------- [STATUS] Processing U-Net output ----------------")
    #input_unet_images have been used to compute the mask of the image
    mask_images = []
    for mask in input_unet_images:
        mask = mask*255
        mask = mask.astype('uint8')
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
        mask = cv.erode(mask, kernel, iterations=3)
        ret, mask = cv.threshold(mask, 1, 255, 0, cv.THRESH_BINARY + cv.THRESH_OTSU)
        mask_images.append(mask)

    segmeted_images = []
    i=0
    for mass in output_unet_images:
        mass = mass * 255
        mass = mass.astype('uint8')
        mass[mask_images[i] == 0] = 0
        segmeted_images.append(mass)
        i+=1
    print("-------------------- [NOTIFY] Outputs processed ----------------------")
    return segmeted_images
