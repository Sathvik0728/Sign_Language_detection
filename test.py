import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D
import os
import warnings
import h5py

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Custom DepthwiseConv2D to load models using 'groups'
def custom_depthwise_conv2d(*args, **kwargs):
    kwargs.pop('groups', None)
    return DepthwiseConv2D(*args, **kwargs)

# Configuration
CONFIDENCE_THRESHOLD = 0.7
MODEL_INPUT_SIZE = 224

# ✅ Corrected Path
model_dir = "E:/SATHVIK/study/Projects/Sign_Language_detection/Model"
model_path = os.path.join(model_dir, "keras_model.h5")
labels_path = os.path.join(model_dir, "labels.txt")

# Create directory if missing
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"[INFO] Created directory: {model_dir}")

def load_model_and_labels():
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found at: {labels_path}")

    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'DepthwiseConv2D': custom_depthwise_conv2d}
        )
        print("[INFO] Model loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    try:
        with open(labels_path, 'r') as f:
            labels = [line.strip() for line in f if line.strip()]
        print(f"[INFO] Loaded {len(labels)} labels: {labels}")
    except Exception as e:
        raise RuntimeError(f"Failed to load labels: {e}")

    if model.output_shape[-1] != len(labels):
        raise ValueError(
            f"Model outputs {model.output_shape[-1]} classes, "
            f"but {len(labels)} labels provided."
        )

    return model, labels

def preprocess_image(image, target_size=(224, 224)):
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)

def initialize_camera(camera_index=0, max_attempts=3):
    for attempt in range(max_attempts):
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"[INFO] Camera opened successfully on index {camera_index}.")
            return cap
        print(f"[WARNING] Failed to open camera on index {camera_index}. Attempt {attempt + 1}/{max_attempts}")
        cap.release()
        camera_index += 1
    raise RuntimeError("Failed to open camera after multiple attempts.")

def main():
    try:
        model, labels = load_model_and_labels()
    except Exception as e:
        print(f"[ERROR] Initialization failed: {e}")
        return

    try:
        cap = initialize_camera(camera_index=0)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    detector = HandDetector(maxHands=1, detectionCon=0.8)
    offset = 20
    imgSize = 300

    while True:
        success, img = cap.read()
        if not success:
            print("[WARNING] Failed to read frame from camera. Retrying...")
            cap.release()
            try:
                cap = initialize_camera(camera_index=0)
            except Exception as e:
                print(f"[ERROR] {e}")
                break
            continue

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            try:
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[
                    max(0, y - offset):min(img.shape[0], y + h + offset),
                    max(0, x - offset):min(img.shape[1], x + w + offset)
                ]
                if imgCrop.size == 0:
                    raise ValueError("Empty hand crop")

                aspect_ratio = h / w
                if aspect_ratio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wGap + wCal] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hGap + hCal, :] = imgResize

                imgInput = preprocess_image(imgWhite)
                prediction = model.predict(imgInput, verbose=0)[0]

                index = np.argmax(prediction)
                confidence = prediction[index]

                if confidence >= CONFIDENCE_THRESHOLD:
                    label_text = f"{labels[index]} ({confidence:.2f})"
                    color = (0, 255, 0)
                else:
                    label_text = f"Uncertain ({confidence:.2f})"
                    color = (0, 0, 255)

                cv2.rectangle(imgOutput, (x - offset, y - offset - 70),
                             (x + 250, y - offset - 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(imgOutput, label_text, (x, y - 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                             (x + w + offset, y + h + offset), color, 4)

                cv2.imshow("Hand Crop", imgCrop)
                cv2.imshow("Processed Input", imgWhite)

                print("\n=== Predictions ===")
                for i, (label, prob) in enumerate(zip(labels, prediction)):
                    print(f"{label:12s}: {prob:.4f}{' *' if i == index else ''}")
                print(f"Selected: {label_text}")

            except Exception as e:
                print(f"[WARNING] Processing error: {e}")
                cv2.putText(imgOutput, "Processing error", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(imgOutput, "No hand detected", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Sign Language Detection", imgOutput)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
