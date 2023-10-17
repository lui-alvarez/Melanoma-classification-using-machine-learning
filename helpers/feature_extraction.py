import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode, skew, kurtosis, entropy

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog

class FeatureExtraction:
    def __init__(self) -> None:
        pass

    
    def calculate_hist_features(self, hist):
        feature_vector = []
        # feature_vector.append(np.mean(hist)) # Mean
        # feature_vector.append(mode(hist)[0][0]) # Mode
        feature_vector.append(np.std(hist)) # Standard deviation
        feature_vector.append(skew(hist)[0]) # Skewness
        feature_vector.append(np.sum(hist**2)) # Energy
        feature_vector.append(entropy(hist, base=2)[0]) # Entropy
        feature_vector.append(kurtosis(hist)[0]) # Kurtosis

        return np.array(feature_vector)
    
    def extract_color_features(self, img):
        b, g, r = cv2.split(img) # Split the image into the B, G, R channels
        n_bins = 256 # Number of bins
        height, width, _ = img.shape
        N = height * width # Number of pixels

        # Calculate normalized histograms for each channel
        hist_r = (cv2.calcHist([r], [0], None, [n_bins], [0, 256])) / N
        hist_g = (cv2.calcHist([g], [0], None, [n_bins], [0, 256])) / N
        hist_b = (cv2.calcHist([b], [0], None, [n_bins], [0, 256])) / N

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_img)

        # Calculate normalized histograms for each channel
        hist_h = (cv2.calcHist([h], [0], None, [n_bins], [0, 256])) / N
        hist_s = (cv2.calcHist([s], [0], None, [n_bins], [0, 256])) / N
        hist_v = (cv2.calcHist([v], [0], None, [n_bins], [0, 256])) / N


        color_features = np.concatenate((
            self.calculate_hist_features(hist_r), 
            self.calculate_hist_features(hist_g), 
            self.calculate_hist_features(hist_b),
            
            self.calculate_hist_features(hist_h), 
            self.calculate_hist_features(hist_s), 
            self.calculate_hist_features(hist_v)
            )
        )

        return color_features


    def extract_glcm_features(self, image):    
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a numpy array")

        if image.dtype != np.uint8:
            # If the image is not 8-bit, convert it to 8-bit
            # important as we set the levels for glcm to 256 for 8-bit images
            image = image.astype(np.uint8)

        # Convert image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute GLCM matrix
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(gray_image, distances, angles, levels=256, symmetric=True, normed=True)
        
        # Extract texture features from GLCM matrix
        features = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        texture_features = np.concatenate([graycoprops(glcm, feature).ravel() for feature in features])
        
        return texture_features
    
    def extract_lbp_features(self, image):
        ''' 
        Extracts the histogram of the LBP image for a certain number of neighbours and radius.

        Args:
            image (numpy.ndarray): the image from which the LBP features will be extracted (cropped ROI)
            P (int): number of neighbours
            R (int): radius
        Output:
            hist (numpy.array): histogram of the LBP image
        '''

        P=[8, 16] 
        R=[1, 2]
        histogram = []

        # Convert image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        for idx, P_value in enumerate(P):

            n_points = P_value * R[idx]

            lbp_image = local_binary_pattern(gray_image, n_points, R[idx], method='uniform')

            # Calculate the histogram of the LBP image
            hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2), density=True)

            # Normalize the histogram
            hist /= np.sum(hist)

            histogram = np.concatenate((histogram, hist))
        
        return histogram

    def extract_hog_features(self, image):
        """
        Extract HOG features from an input image.

        Parameters:
        - image: Input image (grayscale).

        Returns:
        - hog_features: The extracted HOG features.
        """
        # Convert image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        # Define HOG parameters (you can customize these as needed)
        pixels_per_cell = (8, 8)
        cells_per_block = (2, 2)
        orientations = 9

        # Compute HOG features
        hog_features = hog(gray_image, pixels_per_cell=pixels_per_cell,
                           cells_per_block=cells_per_block,
                           orientations=orientations)
        
        return hog_features
        
    def fit(self, image):
        color_features      = self.extract_color_features(image)
        texture_features    = self.extract_glcm_features(image)
        lbp_features        = self.extract_lbp_features(image)
        # hog_features        = self.extract_hog_features(image) # 1 file of 7k samples ended up 24GB size

        all_features = np.concatenate((color_features, texture_features, lbp_features))

        return all_features






    