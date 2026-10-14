import cv2
import numpy as np

class RefinementStep:
    def refine(self, mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if num_labels > 1:
            largest_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            return (labels == largest_idx).astype(np.uint8)
        return mask
