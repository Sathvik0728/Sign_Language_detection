# Sign Language Detection

AI-powered real-time Sign Language Detection system using Computer Vision and Deep Learning. This project detects hand gestures through a webcam and converts sign language gestures into readable text using MediaPipe, TensorFlow, and OpenCV.

---

## Project Overview

This project is designed to help bridge communication between hearing-impaired individuals and others by recognizing hand gestures in real time.

The system captures webcam input, detects hand landmarks using MediaPipe, and predicts sign language gestures using a trained deep learning model.

---

## Features

- Real-time sign language detection
- Webcam-based hand tracking
- AI-powered gesture recognition
- Deep learning prediction model
- Live gesture-to-text conversion
- Hand landmark visualization
- Fast and accurate detection
- Beginner-friendly interface

---

## Dataset Classes

The model is trained on the following gestures:

- Hello
- I Love You
- No
- Okay
- Thank you
- Yes

---

## Tech Stack

- Python
- OpenCV
- MediaPipe
- TensorFlow
- Keras
- NumPy
- Computer Vision
- Deep Learning

---

## Project Structure

```plaintext
Sign_Language_detection/
│
├── Data/
│   ├── Hello/
│   ├── I Love You/
│   ├── No/
│   ├── Okay/
│   ├── Thank_you/
│   └── Yes/
│
├── Model/
│   ├── keras_model.h5
│   └── labels.txt
│
├── data_collection.py
├── test.py
├── README.md
└── requirements.txt
```

---

## Model Files

This project uses:

- `keras_model.h5` → Trained deep learning model
- `labels.txt` → Gesture labels/classes

---

## Installation

### Clone Repository

```bash
git clone https://github.com/Sathvik0728/Sign_Language_detection.git
cd Sign_Language_detection
```

---

### Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

#### Windows

```bash
venv\Scripts\activate
```

#### Mac/Linux

```bash
source venv/bin/activate
```

---

### Install Requirements

```bash
pip install -r requirements.txt
```

---

## Requirements

Create a `requirements.txt` file:

```txt
opencv-python
mediapipe
tensorflow
keras
numpy
```

---

## Run Project

### Collect Dataset

```bash
python data_collection.py
```

---

### Run Detection System

```bash
python test.py
```

---

## How It Works

1. Captures webcam frames using OpenCV
2. Detects hand landmarks using MediaPipe
3. Extracts gesture features
4. Passes data into trained TensorFlow model
5. Predicts sign language gesture
6. Displays output text in real time

---

## Note About Dataset

The dataset folder is not uploaded to GitHub because of large file size limitations.

You can create your own dataset using:

```bash
python data_collection.py
```

---

## Applications

- Assistive communication systems
- AI accessibility tools
- Smart gesture recognition
- Educational learning systems
- Human-computer interaction

---

## Future Improvements

- More gesture classes
- Sentence formation
- Voice output
- Web deployment
- Mobile application
- Better prediction accuracy

---

## Author

Banda Sathvik

GitHub:
https://github.com/Sathvik0728

---

## Repository

https://github.com/Sathvik0728/Sign_Language_detection

---

## License

This project is open-source and available under the MIT License.
