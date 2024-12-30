import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2

# Load the face detector and mask detection models
prototxtPath = r"face_detector/deploy.prototxt"
weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

try:
    model = load_model("mask_detector.keras")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

maskNet = load_model("mask_detector.keras")

def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            if face.shape[0] > 0 and face.shape[1] > 0:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                faces.append(face)
                locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)
# Add tab navigation
tabs = st.tabs(["Detection", "About the Project"])

with tabs[0]:
    # Detection Page
    st.title("Real-Time Face Mask Detection")
    st.write("Click the **Detect** checkbox below to start the webcam and detect face masks in real-time.")

    detect = st.checkbox("Detect")
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    if detect:
        while detect:
            ret, frame = camera.read()
            if not ret:
                st.error("Unable to access the webcam.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            FRAME_WINDOW.image(frame)
    else:
        camera.release()

with tabs[1]:
    # About the Project Page
    st.title("About the Face Mask Detection Project")
    st.write("""
    This project is a real-time face mask detection system using deep learning.
    
    **Key Features:**
    - Detects faces and identifies whether the person is wearing a mask or not.
    - Uses OpenCV for face detection and TensorFlow/Keras for mask classification.
    
    **How It Works:**
    - A pre-trained face detector is used to locate faces in the video feed.
    - Each detected face is passed through a mask classification model, which predicts if a mask is being worn.
    
    **Technologies Used:**
    - Python
    - Streamlit
    - TensorFlow/Keras
    - OpenCV

    **Instructions:**
    - Navigate to the "Detection" tab to start detecting face masks in real-time.
    - Ensure your webcam is enabled and accessible for the app to function correctly.
    
    **Acknowledgments:**
    - The mask detector model was trained on a public dataset of face images with and without masks.
    """)
