import cv2
import numpy as np

def extract_canny_vector(image, low_th=30, high_th=150):
    return cv2.Canny(image, low_th, high_th)
