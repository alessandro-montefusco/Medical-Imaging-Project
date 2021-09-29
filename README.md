___________________________________THE MAIN IDEA__________________________________
Our pipeline is based on the following main steps:
    - First: an SVM classifier is trained thanks to the haralick texture features extracted from the dataset.
            The dataset was first balanced and then agumented in order to well fit the classifier. So, it is divided in
            2 parts: n.582 mass images; n.586 no-mass images.
    - Second: Output images of the classifier are preprocessed in order to enphasize internal structures such as masses.
    - Third: U-Net Neural Network, whose input is the predicted output of the SVM classifier that was pre-processed in
            the previous step. The images are then analyzed in order to extract masses.
    - Fourth: Image Segmentation.

_____________________________________DATASET_____________________________________
The Training Set is composed by:
- "mass" directory: it contains n.483 images
- "no mass" directory: it contains n.606 images

The Test Set contains the 10% of both the "mass" and "no mass" images:
- mass imges n.45
- no mass images n.60

In particular:
- the first 45 images have the mass
- the others 60 images don't have the mass

______________________________A PRIORI INFORMATION_________________________________
- min_perimeter = 60.870057225227356
- max_perimeter = 1043.452877998352
- min_area = 165.5
- max_area = 45060.5

__________________________________ SVM PREDICTION__________________________________
#label 1: MASS
#label 0: NO_MASS

Number of masses: 45
Number of predicted masses:  60
---------------------------------------------------------------
Number of non-masses: 60
Number of predicted non-masses:  45
---------------------------------------------------------------
SVM ACCURACY:  0.8571428571428571
---------------------------------------------------------------
False-positives: 0 - all the masses are well classified
False-negatives: 15 - images with no mass that have been classified as mass


__________________________________ U-NET PREDICTION__________________________________
Test loss: 0.7297657
Test accuracy: 0.8267117

__________________________________ JACCARD INDEXES____________________________________
Average Jaccard index:  0.3975116977331322
--------------------------------
Minimum Jaccard index:  0.0
--------------------------------
Maximum Jaccard index:  1