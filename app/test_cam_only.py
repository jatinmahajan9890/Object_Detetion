import cv2

print("Opening camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: Cannot open camera.")
    exit()

print("Camera opened successfully! Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ ERROR: Can't receive frame (stream end?). Exiting ...")
        break
        
    cv2.imshow('Pure Camera Test', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")