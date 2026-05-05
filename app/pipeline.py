import cv2
from app.detector import run_detector
from app.visualizer import draw_result

def calculate_damage(image, mask):
    # Isolate the surface, blur it to create a smooth clone, and subtract
    isolated_surface = cv2.bitwise_and(image, image, mask=mask)
    smooth_clone = cv2.medianBlur(isolated_surface, 45)
    diff = cv2.absdiff(isolated_surface, smooth_clone)
    
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, defect_map = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)
    
    total_surface_pixels = cv2.countNonZero(mask)
    if total_surface_pixels == 0:
        return 0.0, defect_map
        
    defect_pixels = cv2.countNonZero(defect_map)
    damage_percentage = (defect_pixels / total_surface_pixels) * 100
    
    return damage_percentage, defect_map

def run_pipeline(frame, target_object="auto"):
    # 1. Ask the AI Brain to find the object
    result = run_detector(frame, target_object)
    
    # 2. If it is architecture, calculate the damage!
    is_architecture = result["object"] in ["door", "window", "tiles", "wall"]
    
    if is_architecture and result["mask"] is not None:
        damage_percent, defect_map = calculate_damage(frame, result["mask"])
        
        # Determine the structural condition
        if damage_percent < 1.0:
            result["condition"] = "Perfect / Clean Surface"
        elif damage_percent < 5.0:
            result["condition"] = f"Minor Surface Flaws ({damage_percent:.1f}%)"
        else:
            result["condition"] = f"Major Damage Detected ({damage_percent:.1f}%)"
            
        # Optional: Save the defect map so the visualizer can highlight cracks in red
        result["defect_map"] = defect_map
    else:
        result["condition"] = "N/A"

    # 3. Send everything to the UI Drawer
    output_frame = draw_result(frame, result)
    
    return result, output_frame