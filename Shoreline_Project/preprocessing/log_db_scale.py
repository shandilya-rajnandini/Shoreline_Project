import numpy as np
import cv2

def log_transform(image_array):
    """
    Formulation: sigma_dB = 10 * log10(sigma_raw)
    Converts multiplicative radar backscatter power values into additive decibel footprint matrix.
    """
    img_float = image_array.astype(np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        db_img = 10 * np.log10(np.where(img_float > 0, img_float, 1e-5))
    db_img = np.nan_to_num(db_img, nan=-30.0, posinf=5.0, neginf=-30.0)
    return cv2.normalize(db_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
