# Import required libraries
from flask import Flask, render_template, Response, request
# Flask: for web server, routing, templates, and HTTP handling

from fer.fer import FER
# FER: Facial Emotion Recognition library that uses pre-trained CNNs to detect faces and classify emotions

from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Hugging Face Transformers: to load a pretrained model for text sentiment analysis

import torch
import torch.nn.functional as F
# PyTorch: handles tensors and computations for the text sentiment model

import cv2, time
# OpenCV (cv2): captures video frames from webcam
# time: used for calculating FPS (frames per second)

from collections import deque
# deque: used as a queue to keep track of recent FPS values for smoothing

# ========== FLASK APP INITIALIZATION ==========
app = Flask(__name__)
# Create a Flask app instance — this will handle all web routes and responses

# ========== TEXT SENTIMENT MODEL SETUP ==========
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
# The name of the pretrained Hugging Face model for sentiment classification

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Tokenizer: breaks raw text into tokens and converts to model-readable input IDs

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
# The actual DistilBERT model fine-tuned on sentiment classification (Positive / Negative)

# ========== FACIAL EMOTION DETECTOR CONFIGURATION ==========
emotion_detector = FER(mtcnn=True)
# Create a FER detector instance using MTCNN for more accurate face detection

cap = cv2.VideoCapture(0)
# Open webcam (camera index 0 = default system webcam)

# Set webcam properties (resolution, frame rate)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Define tuning parameters for performance and smoothing
PROCESS_EVERY_N_FRAMES = 5     # Analyze emotions every 5th frame (to save processing)
SMOOTHING = 0.7                # Weight used for exponential smoothing of emotion scores
FRAME_RESIZE_WIDTH = 640       # Resize frames to a fixed width
prev_emotions = {}             # Dictionary to store previous smoothed emotion probabilities
last_detections = []           # List of the most recent emotion detection results
fps_queue = deque(maxlen=30)   # Queue to store recent FPS readings for averaging
prev_time = time.time()        # Used for measuring time between frames (for FPS calculation)

# ========== HELPER FUNCTION FOR SMOOTHING ==========
def smooth(prev, new, alpha):
    """
    Perform exponential smoothing of emotion probabilities:
    new_value = alpha * prev + (1 - alpha) * new
    This makes emotion predictions less jittery.
    """
    return alpha * prev + (1 - alpha) * new


# ========== FRAME GENERATOR FUNCTION (VIDEO STREAM) ==========
def gen_frames():
    """
    Continuously capture webcam frames, detect emotions every few frames,
    draw bounding boxes and labels, encode as JPEG, and yield them
    for MJPEG video streaming in Flask.
    """
    global prev_time
    frame_count = 0  # Counter to know when to process emotions

    while True:
        success, frame = cap.read()  # Capture a frame from the webcam
        if not success:
            break  # Exit loop if webcam read fails

        # Calculate current FPS (frames per second)
        curr_time = time.time()
        fps_queue.append(1 / (curr_time - prev_time))
        avg_fps = sum(fps_queue) / len(fps_queue)  # Average FPS from recent frames
        prev_time = curr_time

        # Resize frame for consistent processing
        h, w = frame.shape[:2]
        scale = FRAME_RESIZE_WIDTH / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        # Process emotion detection every Nth frame to save compute
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            results = emotion_detector.detect_emotions(frame)
            global last_detections
            last_detections = []  # Clear previous detections

            # Loop through all detected faces
            for result in results:
                (x, y, w, h) = result["box"]  # Face bounding box
                emotions = result["emotions"]  # Dict of emotion:probability

                # Smooth the emotion probabilities using exponential averaging
                for k, v in emotions.items():
                    prev_emotions[k] = smooth(prev_emotions.get(k, v), v, SMOOTHING)

                # Pick the emotion with the highest smoothed probability
                top_emotion = max(prev_emotions, key=prev_emotions.get)
                confidence = prev_emotions[top_emotion]

                # Store this face’s detection info for drawing
                last_detections.append((x, y, w, h, top_emotion, confidence))

        # Draw boxes and emotion labels for all detected faces
        for (x, y, w, h, top, conf) in last_detections:
            # Draw green rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Write emotion label + confidence above box
            cv2.putText(frame, f"{top} ({conf:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show FPS on top-left corner
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Encode frame as JPEG image to send to the web browser
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the JPEG bytes with proper multipart headers for live streaming
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Increment frame counter
        frame_count += 1

# ========== ROUTES ==========
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    text = request.form.get('user_text')
    if not text:
        return "Please enter some text."

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
    labels = ["NEGATIVE", "POSITIVE"]
    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()

    return f"Sentiment: {labels[pred]} (Confidence: {confidence*100:.2f}%)"

if __name__ == "__main__":
    app.run(debug=True)