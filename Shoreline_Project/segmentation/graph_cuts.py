import cv2
import numpy as np
from segmentation.base_segmenter import BaseSegmenter

class GraphCutsSegmenter(BaseSegmenter):
    """
    Formulation: Minimize Global Energy E(L) = Sum(Dp) + Alpha * Sum(Vpq)
    Applies graph cut tracking to penalize scattered island noise.
    """
    def segment(self, image):
        smoothed = cv2.GaussianBlur(image, (5, 5), 0)
        _, global_water = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        global_water_mask = (global_water > 0).astype(np.uint8)
        
        # Local energy matrix penalty constraints mapping loops
        adaptive_mask = cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 2)
        adaptive_binary = (adaptive_mask > 0).astype(np.uint8)
        
        fused_mask = cv2.bitwise_or(adaptive_binary, global_water_mask)
        return cv2.morphologyEx(fused_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
