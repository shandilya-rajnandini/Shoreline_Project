import numpy as np
import cv2
from segmentation.base_segmenter import BaseSegmenter

class KMeansSegmenter(BaseSegmenter):
    """
    Formulation: Minimize J = Sum( ||x_i - c_j||^2 )
    Groups gray intensities into distinct clusters using iterative spatial tracking loops.
    """
    def segment(self, image):
        pixel_vals = image.flatten().astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixel_vals, 2, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
        mask = labels.reshape(image.shape).astype(np.uint8)
        if np.argmin(centers) == 1: 
            mask = 1 - mask
        return mask
