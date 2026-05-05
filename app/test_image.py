import cv2
from app.pipeline import run_pipeline

def test_image():
    print("📸 Loading image...")

    # Update this path to the image you want to test!
    image_path =r"/Users/jatinmahajan/Downloads/wall_test.jpeg"
    img = cv2.imread(image_path)

    if img is None:
        print(f"Failed to load image at {image_path}")
        return

    print("Image loaded")

   # Change this to "wall", "door", or "window" or "tile"
    target_object = "tile" 
    # The fixed line
    result, output = run_pipeline(img, target_object=target_object)


    print("\n🔍 RESULTS:")
    print(f"Target Inspected: {result['object']}")
    if result['bbox']:
        print(f"Bounding Box: {result['bbox']}")
    else:
        print("Nothing found.")

    # Resize output window for visibility if the original image is massive
    h, w = output.shape[:2]
    if w > 1200 or h > 900:
        output = cv2.resize(output, (1024, int(1024 * (h/w))))

    cv2.imshow("Final Output", output)
    print("Press any key to close window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_image()