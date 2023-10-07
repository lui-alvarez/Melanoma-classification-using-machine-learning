import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode, skew, kurtosis, entropy

from skimage.feature import graycomatrix, graycoprops

class FeatureExtraction:
    def __init__(self) -> None:
        pass

    def calculate_hist_features(self, hist):
        feature_vector = []
        feature_vector.append(np.mean(hist)) # Mean
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

        color_features = np.concatenate((
            self.calculate_hist_features(hist_r), 
            self.calculate_hist_features(hist_g), 
            self.calculate_hist_features(hist_b))
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
        
    def fit(self, image):
        color_features      = self.extract_color_features(image)
        texture_features    = self.extract_glcm_features(image)

        all_features = np.concatenate((color_features, texture_features))

        return all_features






    