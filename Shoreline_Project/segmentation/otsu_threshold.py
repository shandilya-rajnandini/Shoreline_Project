import cv2
import numpy as np
from segmentation.base_segmenter import BaseSegmenter

class OtsuSegmenter(BaseSegmenter):
    """
    Formulation: Maximize Inter-Class Variance sigma_B^2(t)
    Automatically splits histogram elements into optimal foreground and background bands.
    """
    def segment(self, image):
        _, mask = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask.astype(np.uint8)
