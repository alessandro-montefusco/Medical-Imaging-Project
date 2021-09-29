from utils import data_preprocessing
from utils.jaccard import jaccard
from predictions.SVM_Classifier import SVM_Classifier
from predictions.UNet import UNet
from draw_mass import drawer
import os
import cv2 as cv

############################ PATH DEFINITION ############################
nomass_path = "dataset/images/nomass"
mass_path = "dataset/images/mass"
overlay_path = "dataset/overlay"
test_path = "dataset/test"
mask_path = "dataset/masks"
ground_path = "dataset/groundtruth/groundtruth"
ground_test_path = "dataset/groundtruth/ground_test"
################################   END   ################################

# STEP 1:   Extracting the features from the training set in order to fit the SVM classifier. This step ends with a list of
#           predicted masses (it is also shown the accuracy of the classifier).
classifier = SVM_Classifier(nomass_path, mass_path, overlay_path, mask_path, ground_path, test_path)
classifier.labelling()
classifier.extract_features()
classifier.train_classifier()
predicted_mass, path_predicted_mass = classifier.prediction()

#STEP 2:    Pre-processing of the images to enhance internal structures, before to give them to the Neural Net.
predicted_mass = data_preprocessing.preprocessing(predicted_mass)
predicted_mass = data_preprocessing.cropping(mask_path, predicted_mass, path_predicted_mass)

#STEP 3:    Loading the U-Net model and predicting masses of test set
unet = UNet()
predictions = unet.unet_predict(predicted_mass)

#STEP 4: Segmentation process and final output
segmented_images = drawer.clean_unet_images(predicted_mass, predictions)
outcomes, pred_groundtruth = drawer.my_draw_contours(segmented_images, ground_path, path_predicted_mass)

#STEP 5:    Evaluating performance
jaccard_list = jaccard(pred_groundtruth, path_predicted_mass, ground_test_path)
average = sum(jaccard_list)/len(jaccard_list)
minimum = min(jaccard_list)
maximum = max(jaccard_list)

print("Average Jaccard index: ", average)
print("--------------------------------")
print("Minimum Jaccard index: ", minimum)
print("--------------------------------")
print("Maximum Jaccard index: ", maximum)
