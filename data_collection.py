import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Function to initialize camera with retry mechanism and backend support
def initialize_camera(camera_index=0, max_attempts=3, preferred_backend=cv2.CAP_DSHOW):
    """Initialize camera with retry mechanism and specified backend"""
    backends = [
        (preferred_backend, "Preferred Backend"),
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Microsoft Media Foundation"),
        (cv2.CAP_VFW, "Video for Windows")
    ]

    for backend, backend_name in backends:
        print(f"Trying backend: {backend_name}")
        for attempt in range(max_attempts):
            cap = cv2.VideoCapture(camera_index, backend)
            if cap.isOpened():
                print(f"[INFO] Camera opened with {backend_name} on index {camera_index}.")
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"[INFO] Successfully read a test frame with {backend_name}.")
                    return cap
                else:
                    print(f"[WARNING] Camera opened but failed to read test frame with {backend_name} on index {camera_index}.")
                    cap.release()
            print(f"[WARNING] Failed to open camera with {backend_name} on index {camera_index}. Attempt {attempt + 1}/{max_attempts}")
            camera_index += 1  # Try next index if current fails
    raise RuntimeError("Failed to open camera after multiple attempts with all backends.")

# Initialize camera
try:
    cap = initialize_camera(camera_index=0, preferred_backend=cv2.CAP_DSHOW)
except Exception as e:
    print(f"[ERROR] {e}")
    exit()

# Initialize hand detector
detector = HandDetector(maxHands=1)

# Configuration
offset = 20
imgSize = 300
counter = 0
folder = "E:/SATHVIK/study/Sign_Language_detection/Data"
max_frame_retries = 5  # Maximum retries for frame reading

# Create the output directory if it doesn't exist
if not os.path.exists(folder):
    os.makedirs(folder)
    print(f"[INFO] Created directory: {folder}")

frame_retry_count = 0

while True:
    success, img = cap.read()
    if not success or img is None:
        print("[WARNING] Failed to read frame from camera. Retrying...")
        frame_retry_count += 1
        if frame_retry_count >= max_frame_retries:
            print(f"[ERROR] Failed to read frame after {max_frame_retries} retries. Exiting...")
            break
        cap.release()
        try:
            cap = initialize_camera(camera_index=0, preferred_backend=cv2.CAP_DSHOW)
            frame_retry_count = 0  # Reset retry count on successful initialization
        except Exception as e:
            print(f"[ERROR] {e}")
            break
        continue

    # Reset retry count on successful frame read
    frame_retry_count = 0

    # Debug: Display raw frame before processing
    cv2.imshow("Raw Frame", img)

    # Process the frame for hand detection
    try:
        hands, img = detector.findHands(img)
    except Exception as e:
        print(f"[WARNING] Hand detection failed: {e}")
        continue

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        try:
            # Create a white background
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            # Crop the hand region with boundary checks
            imgCrop = img[
                max(0, y - offset):min(img.shape[0], y + h + offset),
                max(0, x - offset):min(img.shape[1], x + w + offset)
            ]

            if imgCrop.size == 0:
                print("[WARNING] Empty hand crop. Skipping frame.")
                continue

            imgCropShape = imgCrop.shape
            aspectratio = h / w

            if aspectratio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Display the cropped and processed images
            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

        except Exception as e:
            print(f"[WARNING] Processing error: {e}")
            continue

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    # Save image when 's' is pressed
    if key == ord('s'):
        if 'imgWhite' in locals():  # Ensure imgWhite exists before saving
            counter += 1
            timestamp = time.time()
            save_path = f"{folder}/Image_{timestamp}.jpg"
            cv2.imwrite(save_path, imgWhite)
            print(f"[INFO] Saved image {counter}: {save_path}")
        else:
            print("[WARNING] No processed image to save. Make sure a hand is detected.")

    # Exit on ESC key
    if key == 27:  # ESC key
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
