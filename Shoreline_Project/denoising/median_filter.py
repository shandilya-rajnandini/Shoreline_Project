import cv2
def apply_median(img, size=5):
    return cv2.medianBlur(img, size)
