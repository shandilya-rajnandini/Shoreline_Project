import numpy as np
import cv2

def apply_lee_filter(matrix):
    """
    Formulation: W = var_I / (var_I + var_noise) | I_hat = mean + W * (I - mean)
    Preserves structural edge sharpness by calculating local pixel adaptive weights.
    """
    img_f = matrix.astype(np.float32)
    mean = cv2.blur(img_f, (5, 5))
    var = cv2.blur(img_f**2, (5, 5)) - mean**2
    noise_var = np.mean(var) + 1e-5
    weights = var / (var + noise_var)
    return np.clip(mean + weights * (img_f - mean), 0, 255).astype(np.uint8)

def apply_frost_filter(matrix, size=5):
    """
    Formulation: W(i,j) = exp(-a * (sigma/mean) * d(i,j))
    Executes exponential damping functions to smooth flat zones while safeguarding structural glare.
    """
    img_f = matrix.astype(np.float32)
    mean = cv2.blur(img_f, (size, size))
    var = cv2.blur(img_f**2, (size, size)) - mean**2
    
    # Coefficient of variation spatial mapping tracking loop
    cv_matrix = np.where(mean > 0, np.sqrt(np.maximum(0, var)) / mean, 0)
    
    # Calculate geometric pixel coordinates distance array constraints
    y, x = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    dist_grid = np.sqrt(x**2 + y**2)
    
    # Smooth local regions based on adaptive gradient dampening weights
    factor = 0.1 * cv_matrix
    kernel_weights = np.exp(-factor[..., np.newaxis, np.newaxis] * dist_grid)
    kernel_sum = np.sum(kernel_weights, axis=(2, 3)) + 1e-5
    
    # Convolve matrix components manually to preserve fine radar borders
    padded = np.pad(img_f, size//2, mode='edge')
    output = np.zeros_like(img_f)
    for r in range(img_f.shape[0]):
        for c in range(img_f.shape[1]):
            window = padded[r:r+size, c:c+size]
            output[r, c] = np.sum(window * kernel_weights[r, c]) / kernel_sum[r, c]
            
    return np.clip(output, 0, 255).astype(np.uint8)

def apply_kuan_filter(matrix, size=5):
    """
    Formulation: W = (1 - C_u^2 / C_s^2) / (1 + C_u^2)
    Direct multiplicative radar noise modeling for seamless homogeneous water texturing.
    """
    img_f = matrix.astype(np.float32)
    mean = cv2.blur(img_f, (size, size))
    var = cv2.blur(img_f**2, (size, size)) - mean**2
    
    noise_variance = 0.04 # Standard L-band noise multiplier metrics floor calibration
    cs_sq = var / (mean**2 + 1e-5)
    
    w = (1.0 - noise_variance / (cs_sq + 1e-5)) / (1.0 + noise_variance)
    w = np.clip(w, 0.0, 1.0)
    
    return np.clip(mean + w * (img_f - mean), 0, 255).astype(np.uint8)
