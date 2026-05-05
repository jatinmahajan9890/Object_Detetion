import cv2
import numpy as np

def draw_result(image, result):
    output = image.copy()
    target_obj = result.get("object", "unknown")
    mask = result.get("mask")
    bbox = result.get("bbox")
    condition = result.get("condition", "N/A")
    defect_map = result.get("defect_map", None)

    # 🛑 SAFE DEFAULTS: Prevents the UnboundLocalError Crash!
    target_display = "Scanning Scene..."
    condition_color = (255, 255, 255) # White

    # If nothing is detected, show the scanning dashboard
    if target_obj == "unknown" or target_obj is None:
        cv2.rectangle(output, (10, 10), (550, 100), (0, 0, 0), -1)
        cv2.putText(output, "Object Detected: Scanning...", (20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(output, "Condition: Waiting for target...", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return output

    # 1. DRAW ARCHITECTURE (Translucent Green Mask + Sharp Outline)
    if target_obj in ["wall", "door", "window", "tiles"] and mask is not None:
        
        # Highlight actual physical damage in Bright Red
        if defect_map is not None:
            output[defect_map == 255] = [0, 0, 255] 
            
        # Draw the translucent green fill
        colored_overlay = np.zeros_like(image)
        colored_overlay[:] = (0, 255, 0)
        wall_only_overlay = cv2.bitwise_and(colored_overlay, colored_overlay, mask=mask)
        output = cv2.addWeighted(output, 1.0, wall_only_overlay, 0.3, 0)

        # Draw the sharp green geometric outline
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
        
        target_display = f"{target_obj.capitalize()} (Segmented)"
        
        if "Minor" in condition: condition_color = (0, 255, 255) # Yellow
        elif "Major" in condition: condition_color = (0, 0, 255) # Red
        else: condition_color = (0, 255, 0) # Green

    # 2. DRAW EVERYDAY OBJECTS (Bounding Box only)
    elif bbox is not None:
        x, y, bw, bh = bbox
        cv2.rectangle(output, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
        target_display = f"{target_obj.capitalize()} (Detected)"
        condition_color = (255, 255, 255) 

    # 3. DRAW THE STATIC BLACK DASHBOARD
    cv2.rectangle(output, (10, 10), (550, 100), (0, 0, 0), -1)

    cv2.putText(output, f"Object Detected: {target_display}", (20, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.putText(output, f"Condition: {condition}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, condition_color, 2)

    return output