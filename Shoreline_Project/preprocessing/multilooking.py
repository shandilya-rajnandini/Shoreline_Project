import numpy as np
from skimage.util import view_as_blocks
def apply_multilooking(image, look_window=(2, 2)):
    pad_h = (look_window[0] - image.shape[0] % look_window[0]) % look_window[0]
    pad_w = (look_window[1] - image.shape[1] % look_window[1]) % look_window[1]
    padded_img = np.pad(image, ((0, pad_h), (0, pad_w)), mode='edge')
    blocks = view_as_blocks(padded_img, look_window)
    return np.mean(blocks, axis=(2, 3))
