import cv2
import numpy as np

def analyze_defects(image, result):
    mask = result.get("mask")
    target_obj = result.get("object", "unknown")

    # If it is an everyday object or we have no mask, skip the heavy math
    if target_obj not in ["wall", "door", "window", "tiles"] or mask is None:
        return "N/A (General Object)"

    # 1. Isolate surface
    isolated_surface = cv2.bitwise_and(image, image, mask=mask)
    
    # 2. Digital Sandpaper (Blur)
    smooth_clone = cv2.medianBlur(isolated_surface, 45)
    
    # 3. Anomaly Subtraction
    diff = cv2.absdiff(isolated_surface, smooth_clone)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, defect_map = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)
    
    # 4. Calculate Percentage
    total_surface_pixels = cv2.countNonZero(mask)
    if total_surface_pixels == 0:
        return "Perfect / No Damage"
        
    defect_pixels = cv2.countNonZero(defect_map)
    damage_percent = (defect_pixels / total_surface_pixels) * 100
    
    # 5. Format string for Visualizer
    if damage_percent < 1.0:
        return "Perfect / No Damage"
    elif damage_percent < 5.0:
        return f"Minor Surface Flaws ({damage_percent:.1f}%)"
    else:
        return f"Major Damage Detected ({damage_percent:.1f}%)"