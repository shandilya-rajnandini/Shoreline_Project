import cv2
import numpy as np
from scipy.ndimage import center_of_mass, label
from segmentation.base_segmenter import BaseSegmenter

class RegionGrowingSegmenter(BaseSegmenter):
    """
    Formulation: |I(x_n, y_n) - mu_region| <= Delta
    Grows connected, uniform masks out from programmatic deep-water core seeds.
    """
    def segment(self, image):
        h, w = image.shape
        _, water_est = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        labeled, num_feats = label(water_est)
        if num_feats > 0:
            cy, cx = center_of_mass(labeled == (np.argmax(np.bincount(labeled.flat)[1:]) + 1))
            seed = (int(cx), int(cy))
        else:
            seed = (w // 2, h // 2)
        output_blob = image.copy()
        cv2.floodFill(output_blob, np.zeros((h + 2, w + 2), np.uint8), seed, 255, loDiff=15, upDiff=15)
        return (output_blob == 255).astype(np.uint8)
