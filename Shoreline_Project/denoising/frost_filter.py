import numpy as np
from scipy.ndimage import uniform_filter
def apply_frost_filter(img, win_size=5):
    return uniform_filter(img, size=win_size)
