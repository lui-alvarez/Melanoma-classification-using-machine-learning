#### Summary of 1910.03910v1 (only Abstract, Intro, Preprocessing, Data Augmentation sub-section from DL models section)
Same datasets we used: HAM10000, BCN_20000, 
Largely rely on established methods for skin lesion classifcation including loss balancing, heavy data augmentation, pre-trained, state-of-the-art CNNs and extensive ensembling [9,10]. 
Address data variability by applying a color constancy algorithm and a cropping algorithm to deal with raw, uncropped dermoscopy images.

- HAM10000:
600x450 centered and cropped around the region
dataset curators applied histogram corrections to some images

- BCN_20000:
1024x1024
Challengin as many images are uncropped and lesions in difficult and uncommon locations are present

- MSK Dataset:
MSK dataset contains images with various sizes.

We might not be able to do this - 
For internal evaluation we split the main training dataset into ve folds. The dataset contains multiple images of the same lesions. Thus, we ensure that all images of the same lesion are in the same fold.

- Preprocessing:
1. Cropping strategy for uncropped dermoscopy images. This is to deal with the uncropped images which has large, black areas. Steps for this implementation from the paper:
1.1. Binarize the images with a very low threshold, such that the entire dermoscopy field of view is set to 1
1.2. We find the center of mass and the major and minor axis of an ellipse that has the same second central moments as the inner area.
1.3. Based on these values we derive a rectangular bounding box for cropping that covers the relevant field of view. 

2. How to determine the necessecity for cropping?
2.1 We automatically determine the necessecity for cropping based on a heuristic that tests whether the mean intensity inside the bounding box is substantially different from the mean intensity outside of the bounding box.
2.2 Manual inspection showed that the method was robust. In the training set, 6226 were automatically cropped.

I implemented something similar in the breast segmentation that could be similar -> binarize -> find contours -> find the largest contour -> use opencv to draw the bounding box -> crop the image using that bounding box.
Worth trying the paper implementation maybe due to diversity in the dataset (not all has black bg), it might be good to use their approach.

3. They applied Shades of Gray color constancy method with Minkowski norm p=6, following the last years winner. This is particular important as the datasets used for training differ alot.

4. They resize the larger images in the datasets. They took the HAM10000 resolution as a reference and resize all images' longer side to 600 pixels while preserving the aspect ratio.

- CNN Data Augmentation:
Before feeding the images to the networks, we perform extensive data augmentation. We use random brightness and contrast
changes, random clipping, random rotation, random scaling (with appropriate padding/cropping), and random shear. Furthermore, we use CutOut [7] with one hole and a hole size of 16. We tried to apply the AutoAugment v0 policy, however, we did not observe better performance.


Notes for DL later can be good from this paper, they used focal loss and other approaches that were useful.