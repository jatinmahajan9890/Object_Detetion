import cv2
import numpy as np

def get_texture_score(roi):
    if roi is None or roi.size == 0: return 0
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray_roi, cv2.CV_64F).var()

def check_surface_integrity(roi):
    """Detects if the surface is 'broken' by defects or uneven paint."""
    if roi is None or roi.size == 0: return 100
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Calculate the standard deviation of the Hue and Value (Brightness)
    _, stddev = cv2.meanStdDev(hsv)
    # Higher variance means more 'uneven' paint or defects
    return np.mean(stddev)