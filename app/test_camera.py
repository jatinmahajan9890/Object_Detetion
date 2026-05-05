import cv2
import time
import threading
from imutils.video import VideoStream
from app.pipeline import run_pipeline
from app.visualizer import draw_result

# --- SHARED MEMORY (The Clipboard between the two universes) ---
latest_frame = None
latest_result = None
keep_running = True
ai_is_thinking = False

def ai_worker(target_object):
    """This runs completely in the background, independent of the camera feed."""
    global latest_frame, latest_result, keep_running, ai_is_thinking
    
    while keep_running:
        # If we have a frame and the AI isn't currently busy...
        if latest_frame is not None and not ai_is_thinking:
            ai_is_thinking = True
            
            # 1. Grab a copy of the exact frame the camera is looking at right now
            frame_to_process = latest_frame.copy()
            
            # 2. Run the heavy YOLO math and Anomaly Subtraction
            result_dict, _ = run_pipeline(frame_to_process, target_object)
            
            # 3. Paste the results onto the shared clipboard
            latest_result = result_dict
            
            ai_is_thinking = False
        else:
            # Sleep for 10 milliseconds to prevent the background thread from burning your CPU
            time.sleep(0.01) 

def start_live_feed(target_object="auto"):
    global latest_frame, latest_result, keep_running

    print(f"📷 Starting True-Async Camera Feed for: {target_object.upper()}")
    print("Press 'q' on your keyboard to quit the camera.")

    # 1. Start the Background AI Thread
    ai_thread = threading.Thread(target=ai_worker, args=(target_object,))
    ai_thread.daemon = True
    ai_thread.start()

    # 2. Start the physical camera sensor (asking the hardware for a smaller size right away)
    vs = VideoStream(src=0, resolution=(640, 480)).start()
    time.sleep(2.0) # Let the sensor warm up

    while True:
        # 3. Pull the absolute newest frame
        frame = vs.read()
        if frame is None:
            time.sleep(0.1)
            continue
            
        frame = cv2.resize(frame, (640, 480))
        
        # 4. Share this frame with the background AI
        latest_frame = frame 

        # 5. Instantly draw whatever the AI's last result was onto this fresh frame
        if latest_result is not None:
            display_frame = draw_result(frame, latest_result)
        else:
            # The AI takes a second to load the first frame
            display_frame = frame.copy()
            cv2.putText(display_frame, "Waking up AI Engine...", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 6. Show the frame instantly!
        cv2.imshow("Civil-Eye: Live Inspector", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Shutting down system...")
            break

    # Clean up both threads safely
    keep_running = False
    ai_thread.join(timeout=1.0)
    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    TARGET = "auto" 
    start_live_feed(TARGET)