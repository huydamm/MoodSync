from flask import Flask, render_template, Response, request
from fer.fer import FER
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import cv2, time
from collections import deque

app = Flask(__name__)

# ========== TEXT SENTIMENT MODEL ==========
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# ========== EMOTION DETECTOR CONFIG ==========
emotion_detector = FER(mtcnn=True)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

PROCESS_EVERY_N_FRAMES = 5
SMOOTHING = 0.7
FRAME_RESIZE_WIDTH = 640
prev_emotions = {}
last_detections = []
fps_queue = deque(maxlen=30)
prev_time = time.time()

def smooth(prev, new, alpha):
    return alpha * prev + (1 - alpha) * new

def gen_frames():
    global prev_time
    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        curr_time = time.time()
        fps_queue.append(1 / (curr_time - prev_time))
        avg_fps = sum(fps_queue) / len(fps_queue)

        h, w = frame.shape[:2]
        scale = FRAME_RESIZE_WIDTH / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        prev_time = curr_time

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            results = emotion_detector.detect_emotions(frame)
            global last_detections
            last_detections = []

            for result in results:
                (x, y, w, h) = result["box"]
                emotions = result["emotions"]
                for k, v in emotions.items():
                    prev_emotions[k] = smooth(prev_emotions.get(k, v), v, SMOOTHING)
                top_emotion = max(prev_emotions, key=prev_emotions.get)
                confidence = prev_emotions[top_emotion]
                last_detections.append((x, y, w, h, top_emotion, confidence))

        for (x, y, w, h, top, conf) in last_detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{top} ({conf:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        frame_count += 1

# ========== ROUTES ==========
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/combined_emotion', methods=['POST'])
def combined_emotion():
    text = request.form.get('user_text', '')

    # --- TEXT SENTIMENT ---
    if text.strip():
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)[0].numpy()
        text_probs = {'negative': float(probs[0]), 'positive': float(probs[1])}
    else:
        text_probs = {'negative': 0.5, 'positive': 0.5}

    # --- VISION EMOTION ---
    success, frame = cap.read()
    if not success:
        face_probs = {'happy': 0.0, 'sad': 0.0, 'neutral': 1.0}
    else:
        faces = emotion_detector.detect_emotions(frame)
        if faces:
            face_probs = faces[0]['emotions']
        else:
            face_probs = {'happy': 0.0, 'sad': 0.0, 'neutral': 1.0}

    # --- Determine dominant facial emotion ---
    vision_emotion = max(face_probs, key=face_probs.get)
    vision_conf = face_probs[vision_emotion]

    # --- Weighted fusion ---
    vision_weight = 0.3
    text_weight = 0.7

    combined_positive = text_probs['positive']
    combined_negative = text_probs['negative']

    # Only modify if vision emotion is not neutral
    if vision_emotion != 'neutral':
        if vision_emotion in ['happy', 'surprise']:
            combined_positive = (
                text_weight * text_probs['positive'] +
                vision_weight * vision_conf
            )
            combined_negative = (
                text_weight * text_probs['negative'] +
                vision_weight * (1 - vision_conf)
            )
        elif vision_emotion in ['sad', 'angry', 'fear', 'disgust']:
            combined_positive = (
                text_weight * text_probs['positive'] +
                vision_weight * (1 - vision_conf)
            )
            combined_negative = (
                text_weight * text_probs['negative'] +
                vision_weight * vision_conf
            )

    total = combined_positive + combined_negative
    if total == 0:
        total = 1
    combined_positive /= total
    combined_negative /= total

    # --- Final classification ---
    if combined_positive > combined_negative:
        emotion = "POSITIVE"
        conf = combined_positive * 100
    else:
        emotion = "NEGATIVE"
        conf = combined_negative * 100

    return {'emotion': emotion, 'confidence': conf}

if __name__ == "__main__":
    app.run(debug=True)
