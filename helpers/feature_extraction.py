import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode, skew, kurtosis, entropy
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
        n_bins = 255 # Number of bins
        height, width, _ = img.shape
        N = height * width # Number of pixels

        # Calculate normalized histograms for each channel
        hist_r = (cv2.calcHist([r], [0], None, [n_bins], [0, 256]))/N
        hist_g = (cv2.calcHist([g], [0], None, [n_bins], [0, 256]))/N
        hist_b = (cv2.calcHist([b], [0], None, [n_bins], [0, 256]))/N

        color_features = np.concatenate((self.calculate_hist_features(hist_r), self.calculate_hist_features(hist_g), self.calculate_hist_features(hist_b)))

        return color_features

    