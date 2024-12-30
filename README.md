# Face Mask Detection (Local Version)

This project is a real-time face mask detection system that uses deep learning and a webcam for detecting whether a person is wearing a face mask or not. It uses OpenCV for face detection and TensorFlow/Keras for mask classification.

## Prerequisites

Before running this project, ensure you have the following installed:

- **Python 3.12+**
- **pip** for installing dependencies.

## Setup and Installation

1. Clone the repository:
   
   git clone https://github.com/TharunyaSundar/FaceMaskDetection-using-Image-capture.git
   cd FaceMaskDetection-using-Image-capture
   

2. **Create a virtual environment (optional but recommended):**
   
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
  

3. **Install the required dependencies:**
   Install the dependencies listed in the `requirements.txt`:

   pip install -r requirements.txt
   

4. **Download the pre-trained models:**
   This project requires the face detection and mask detection models. Ensure the following files are present:
   - **Face detection model:**
     - `deploy.prototxt` (Face detector architecture)
     - `res10_300x300_ssd_iter_140000.caffemodel` (Pre-trained weights for face detection)
   - **Mask detection model:**
     - `mask_detector.keras` (Trained model for mask detection)

   Place these files in the correct directory (e.g., `face_detector/` for the face detector files and in the root directory for the mask detection model).

## Running the Project Locally

1. **Run the Streamlit app:**
   To start the face mask detection, use the following command:
   
   streamlit run app.py
  

2. **Start detecting masks:**
   - Once the app is running, open it in your browser.
   - You will see a webcam feed with a checkbox to start detecting faces and masks.
   - The system will highlight faces and display whether the person is wearing a mask or not with confidence percentages.

## About the Project

### Why Image Capture Over Video Feed?
While this project works with real-time video processing in a local environment, I have implemented an image capture mechanism for the deployed version of the app. This was done to improve performance and allow better user control over the input. Capturing a still image helps the system to process and predict mask detection in a more controlled manner, which is especially useful in browser-based deployments where video streaming might have limitations.

### How It Works:
1. **Face Detection:** A pre-trained face detection model (based on OpenCV's DNN module) is used to detect faces from the webcam feed.
2. **Mask Detection:** Each detected face is passed through the trained mask detection model (a Keras model). The model predicts whether the person is wearing a mask or not.
3. **Real-Time Feedback:** The results are displayed in real-time on the webcam feed, with bounding boxes around detected faces and labels for mask/no mask along with the prediction confidence.

## Acknowledgments
- The mask detection model was trained on a public dataset of face images with and without masks.
- OpenCV for face detection and TensorFlow/Keras for mask classification.
- Streamlit for providing an easy interface for real-time applications.
