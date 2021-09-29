import cv2 as cv
import os
import mahotas as mt
import sklearn.svm as skl
from sklearn.metrics import accuracy_score
from utils import dataset_preprocessing as dsp
from utils import utilities as ut


class SVM_Classifier:

    def __init__(self, nomass_path, mass_path, overlay_path, mask_path, ground_path, test_path):
        # Create the classifier
        print("-------------------- [STATUS] Creating the classifier ----------------")
        self._nomass_path = nomass_path
        self._mass_path = mass_path
        self._overlay_path = overlay_path
        self._mask_path = mask_path
        self._ground_path = ground_path
        self._test_path = test_path
        #Inizialization of the SVM classifier
        self._my_svm = skl.SVC(C=100, kernel='rbf', gamma='auto', verbose=False, max_iter=-1)
        self._nomass_images = os.listdir(nomass_path)
        # load the overlay dataset
        self._overlay_images = os.listdir(overlay_path)
        # load the mask dataset
        self._mask_images = os.listdir(mask_path)
        # load the groundtruth
        self._ground_images = os.listdir(ground_path)
        # empty list to hold feature vectors and train labels
        self._train_features = []
        self._train_labels = []
        print("-------------------- [NOTIFY] Classifier created ---------------------")

    def _texture_features(self,image):
        '''
        The function extract the Haralick texture features from the current image.
        :param image: the image on which Haralick is computed.
        :return: a list of features of the current image.
        '''
        # calculate haralick texture features for 4 types of adjacency
        textures = mt.features.haralick(image)
        # take the mean of it and return it
        ht_mean = textures.mean(axis=0)
        return ht_mean

    def _evaluate_prediction(self,y_pred, y_true):
        '''
        The function evaluates the performance of the SVM classifier.
        :param y_pred: the list of labels predicted from the SVM classifier.
        :param y_true: the list of true labels associated to the Test Set.
        :return: the accuracy of the SVM classifier.
        '''
        print("===========================================")
        predicted_mass = 0
        predicted_nomass = 0
        for y in y_pred:
            if y == 0:
                predicted_nomass +=1
            else:
                predicted_mass +=1
        print("Number of masses: 45")
        print("Number of predicted masses: ", predicted_mass)
        print("===========================================")
        print("Number of non-masses: 60")
        print("Number of predicted non-masses: ", predicted_nomass)
        print("===========================================")
        print("SVM ACCURACY: ", accuracy_score(y_true, y_pred))
        print("===========================================")

    def labelling(self, labelling=False):
        '''
        The function computes the dataset preprocessing step:
            - remove irrelevant brackground from all the images;
            - divide images into classes (mass / no-mass);
            - balance data for the classifier;
            - augumentation of the dataset for better accuracy.
        :param labelling: the boolean condition: if True the dataset preprocessing must be executed.
        '''
        if(labelling):
            dsp.cleaning(self._mask_path, self._mask_images, self._nomass_path, self._nomass_images)
            dsp.createClasses(self._nomass_path, self._nomass_images, self._overlay_images, self._mass_path)
            dsp.balancing(self._mass_path)
            dsp.augmentation(self._mass_path, self._nomass_path, self._nomass_images)

    def extract_features(self):
        '''
        The function extracts all the features from the images in the Training Set and creates
        the respective labels for each image. It also saves the files with features and labels.
        '''
        mass_images = os.listdir(self._mass_path)
        count_training = 1
        if not ut.check_file(): #check if the features have been already extracted
            print("-------------------- [STATUS] Extracting Haralick textures -----------")
            for mass in mass_images:
                # read the training image
                image = cv.imread(self._mass_path + "\\" + mass, cv.IMREAD_GRAYSCALE)
                self._train_labels.append(int(1))
                # extract haralick texture from the image
                features = self._texture_features(image)
                # append the feature vector and label
                self._train_features.append(features)
                print("Extracting features from image number " + str(count_training))
                count_training += 1
            for nomass in self._nomass_images:
                image = cv.imread(self._nomass_path + "\\" + nomass, cv.IMREAD_GRAYSCALE)
                self._train_labels.append(int(0))
                # extract haralick texture from the image
                features = self._texture_features(image)
                # append the feature vector and label
                self._train_features.append(features)
                print("Extracting features from image number " + str(count_training))
                count_training += 1
            print("-------------------- [NOTIFY] Features extracted ---------------------")
            ut.store(self._train_features,self._train_labels)
        else:
            print("-------------------- [STATUS] Loading features and labels ------------")
            self._train_features,self._train_labels = ut.load()

    def train_classifier(self):
        '''
        The function calls the fit function of the SVM classifier.
        '''
        # Fit the training data and labels
        print ("-------------------- [STATUS] Fitting the model ----------------------")
        self._my_svm.fit(self._train_features, self._train_labels)
        print("-------------------- [NOTIFY] Model fitted ---------------------------")

    def prediction(self, tot_test_images=105, num_mass_test=45):
        '''
        The function extract the features from each mammogram image in the Test Set and calls
        the predict function of the SVM classifier. It also creates a list with the predicted
        labels.
        :param tot_test_images: the total number of images in the Test Set. 105 is the default
        value based on our Test Set.
        :param num_mass_test: the total number of mammogram images with masses. 45 is the
        default value based on our Test Set.
        :return: a tuple of lists: the predict images and the path of these images.
        '''
        print("-------------------- [STATUS] Classifier prediction ------------------")
        # load the test set
        test_images = os.listdir(self._test_path)

        # Loop over the test images
        count_test = 1
        y_pred = []
        y_true = []
        predicted_mass = []
        path_predicted_mass = []
        for test in test_images:
            image = cv.imread(self._test_path + "\\" + test)
            # convert to grayscale
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            # extract haralick texture from the image
            features = self._texture_features(gray)

            # evaluate the model and predict label
            prediction = int(self._my_svm.predict(features.reshape(1, -1))[0])
            if prediction:
                predicted_mass.append(image)
                path_predicted_mass.append(test)
            print("------------------------------")
            # 0 means noMass, while 1 means mass
            print("Prediction of image n. ", str(count_test), ": ", prediction)
            y_pred.append(prediction)

            count_test += 1

        # Evaluate SVM prediction
        for i in range(tot_test_images):
            if i < num_mass_test:
                y_true.append(int(1))
            else:
                y_true.append(int(0))

        self._evaluate_prediction(y_pred, y_true)
        return predicted_mass, path_predicted_mass
