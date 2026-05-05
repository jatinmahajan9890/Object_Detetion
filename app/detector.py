from ultralytics import YOLO
import cv2
import numpy as np
import os
import time

print("🧠 Loading Models...")
seg_model = YOLO("yolov8n-seg.pt")      
# SPEED HACK: Dropped from 'm' to 's' for significantly faster real-time edge processing
world_model = YOLO("yolov8s-worldv2.pt")  

# --- THE VAULT SETUP (Data Flywheel) ---
VAULT_DIR = "backend_vault/unknown_scans"
os.makedirs(VAULT_DIR, exist_ok=True)
last_save_time = 0  # Global variable for the rate limiter
# ---------------------------------------

# 🌍 THE FOCUSED INDOOR DICTIONARY
EVERYDAY_OBJECTS = [
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", 
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", 
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", 
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", 
    "pen", "light", "bottle", "fan", "table"
]

def run_detector(image, target="auto"):
    global last_save_time
    target = target.lower().strip()
    h, w = image.shape[:2]
    
    if target == "auto":
        arch_targets = ["door", "window", "tile floor", "hardwood floor", "wall", "room wall"]
        all_targets = arch_targets + EVERYDAY_OBJECTS
        
        world_model.set_classes(all_targets)
        # SPEED HACK: Using Apple Silicon GPU (mps) and smaller image size
        auto_results = world_model(image, imgsz=320)[0]
        
        best_arch_conf = 0
        best_daily_conf = 0
        best_unknown_conf = 0
        
        arch_target = None
        daily_target = None
        unknown_bbox = None
        
        for box in auto_results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = world_model.names[cls_id].lower()
            
            if label in arch_targets or "floor" in label:
                if conf > best_arch_conf and conf > 0.15: 
                    best_arch_conf = conf
                    if "floor" in label: arch_target = "tiles"
                    elif "wall" in label: arch_target = "wall"
                    else: arch_target = label
                    
            elif label in EVERYDAY_OBJECTS:
                if conf > best_daily_conf and conf > 0.40:
                    best_daily_conf = conf
                    daily_target = label
                    
            # 🚨 THE NEW TRAP: If it's not architecture and not everyday, but >50% confident it's *something*
            else:
                if conf > best_unknown_conf and conf > 0.50:
                    best_unknown_conf = conf
                    unknown_bbox = box.xyxy[0]

        # 📸 THE SILENT CROPPER (Runs if we found an unknown and 3 seconds have passed)
        current_time = time.time()
        if arch_target is None and daily_target is None and unknown_bbox is not None:
            if current_time - last_save_time > 3.0: # 3-second cooldown
                x1, y1, x2, y2 = map(int, unknown_bbox)
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                
                cropped_img = image[y1:y2, x1:x2]
                
                if cropped_img.size > 0:
                    filename = os.path.join(VAULT_DIR, f"unknown_{int(current_time)}.jpg")
                    cv2.imwrite(filename, cropped_img)
                    print(f"📥 [DATA FLYWHEEL] Unknown object caught & saved: {filename}")
                    last_save_time = current_time
                    
        if arch_target is not None: detected_target = arch_target
        elif daily_target is not None: detected_target = daily_target
        else: detected_target = "unknown"
            
        target = detected_target 

    # SPEED HACK applied to the segmentation mask
    seg_results = seg_model(image, device="mps", imgsz=320)[0]
    background_mask = np.ones((h, w), dtype=np.uint8) * 255
    if seg_results.masks is not None:
        for seg in seg_results.masks.xy:
            pts = np.array(seg, dtype=np.int32)
            cv2.fillPoly(background_mask, [pts], 0)
    kernel = np.ones((25, 25), np.uint8)
    background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_CLOSE, kernel)

    if target in ["door", "window"]:
        search_terms = [target, f"front {target}", f"wooden {target}", f"glass {target}"]
        if target == "window": search_terms.append("window frame")
        world_model.set_classes(search_terms)
        
        # SPEED HACK applied here as well
        world_results = world_model(image, imgsz=320)[0]
        
        best_conf = 0
        bbox = None
        for box in world_results.boxes:
            conf = float(box.conf[0])
            if conf > best_conf: 
                best_conf = conf
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                bbox = (x1, y1, x2-x1, y2-y1) 
        
        if bbox is not None and bbox[2] > 0 and bbox[3] > 0:
            x, y, bw, bh = bbox
            if target == "window":
                roi = image[y:y+bh, x:x+bw]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray_roi, 30, 100)
                edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                clean_mask = np.zeros((h, w), dtype=np.uint8)
                if len(contours) > 0:
                    largest = max(contours, key=cv2.contourArea)
                    inner_x, inner_y, inner_w, inner_h = cv2.boundingRect(largest)
                    cv2.rectangle(clean_mask, (x + inner_x, y + inner_y), (x + inner_x + inner_w, y + inner_y + inner_h), 255, -1)
                else:
                    cv2.rectangle(clean_mask, (x, y), (x+bw, y+bh), 255, -1)
                clean_mask = cv2.bitwise_and(clean_mask, background_mask)
                fx, fy, fbw, fbh = cv2.boundingRect(clean_mask)
                return {"object": target, "bbox": (fx, fy, fbw, fbh), "mask": clean_mask}
            else:
                gc_mask = np.zeros((h, w), np.uint8)
                bgdModel, fgdModel = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
                cv2.grabCut(image, gc_mask, bbox, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
                final_mask = np.where((gc_mask == 1) | (gc_mask == 3), 255, 0).astype('uint8')
                contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    clean_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(clean_mask, [max(contours, key=cv2.contourArea)], -1, 255, -1)
                    clean_mask = cv2.bitwise_and(clean_mask, background_mask)
                    rx, ry, rbw, rbh = cv2.boundingRect(clean_mask)
                    return {"object": target, "bbox": (rx, ry, rbw, rbh), "mask": clean_mask}
        return {"object": target, "bbox": None, "mask": None} 

    elif target == "wall":
        final_mask = background_mask.copy()
        cv2.rectangle(final_mask, (0, int(h * 0.70)), (w, h), 0, -1)
        cv2.rectangle(final_mask, (0, 0), (w-1, h-1), 0, 1)
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            bbox = cv2.boundingRect(max(contours, key=cv2.contourArea))
            if ((bbox[2] * bbox[3]) / (w * h)) >= 0.10:
                return {"object": target, "bbox": bbox, "mask": final_mask}
        return {"object": "unknown", "bbox": None, "mask": None}

    elif target in ["tile", "tiles", "planks", "wood"]:
        final_mask = np.zeros((h, w), dtype=np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur1 = cv2.GaussianBlur(gray, (7, 7), 0)
        thresh1 = cv2.adaptiveThreshold(blur1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
        grid1 = cv2.dilate(thresh1, np.ones((5, 5), np.uint8), iterations=1)
        cv2.rectangle(grid1, (0, 0), (w-1, h-1), 255, 3)
        contours1, _ = cv2.findContours(cv2.bitwise_not(grid1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_pieces = [c for c in contours1 if (w * h * 0.002) < cv2.contourArea(c) < (w * h * 0.95)]
        if len(valid_pieces) > 0:
            for c in valid_pieces: cv2.drawContours(final_mask, [c], -1, 255, -1)
            return {"object": "tiles", "bbox": cv2.boundingRect(final_mask), "mask": final_mask}
        return {"object": "unknown", "bbox": None, "mask": None}

    elif target in EVERYDAY_OBJECTS:
        world_model.set_classes([target])
        # SPEED HACK applied
        generic_results = world_model(image, imgsz=320)[0]
        best_conf, bbox = 0, None
        for box in generic_results.boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = (x1, y1, x2-x1, y2-y1)
        if bbox is not None:
            return {"object": target, "bbox": bbox, "mask": None}

    return {"object": "unknown", "bbox": None, "mask": None}