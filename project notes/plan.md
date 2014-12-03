
# Overall Theme

* Improving a limited dataset by augmenting it with "similar" data, hoping that the underlying feature representations can help it. 

# Presentation

* Likely we will not have perfect results by Tuesday
* After presentation, try to bump up the accuracy
* If a method doesn't work, understand why it doesn't work
* Describe future directions
    - Motion data in addition to flat images

## To Add

* Specific prior art

# Techniques

* Haar cascade
    - Or different cascade, like HOG or binary pattern
    - Multiple haar cascades
* Hog features + SVM
* Gabor features + SVM
* SIFT + SVM
* Other image features + SVM
* Caffe Feature Extraction + SVM

# Variations

* Grayscale on images
* Webcam images alone
* Webcam images + similar bear images
* Existing datasets + augmented dataset
* Existing methods + transforms (like reflection, translation) 

+ RBF SVM on conv5 features

# Methods



# Datasets

Potentially store everything in separate directories. 
For example, separate the images of bears based on whether they're from the Internet or from the camera framegrabs, and from the framegrabs whether they're cropped or not, for example.

Test data and training data separated mainly by a CSV file.

## Negative Example Sources

* Visually similar reverse image search on Google

# Evaluation

## Object Detection

mAP

Jaccard Similarity

F-Measure
