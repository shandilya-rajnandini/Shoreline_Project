import cv2
import numpy as np

def apply_douglas_peucker(edge_matrix, tolerance=2.0):
    if np.sum(edge_matrix > 0) == 0: 
        return edge_matrix
    pts = np.argwhere(edge_matrix > 0)
    epsilon = 0.02 * cv2.arcLength(pts, closed=False)
    approx = cv2.approxPolyDP(pts, epsilon, closed=False)
    canvas = np.zeros_like(edge_matrix)
    for p in approx: 
        canvas[p, p] = 255
    return canvas
