import numpy as np
from segmentation.base_segmenter import BaseSegmenter
class FCMSegmenter(BaseSegmenter):
    def segment(self, image):
        return (image > np.mean(image)).astype(np.uint8)
