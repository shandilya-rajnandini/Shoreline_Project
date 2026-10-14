from scipy.ndimage import uniform_filter
def apply_kuan_filter(img, size=5):
    return uniform_filter(img, size=size)
