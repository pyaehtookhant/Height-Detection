from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
import threading

app = Flask(__name__)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Constants
A4_REAL_HEIGHT = 0.297  # A4 height in meters (297mm)
A4_ASPECT_RATIO = 210 / 297
ASPECT_TOLERANCE = 0.15

# Shared resources
height_data = {"height": "N/A", "is_active": False}
camera = None
camera_lock = threading.Lock()

def detect_a4_paper(frame):
    """Detect A4 paper using contour analysis"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 60, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
        
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / h
            if abs(aspect - A4_ASPECT_RATIO) < ASPECT_TOLERANCE:
                return (x, y, x + w, y + h), h
    return None, None

def calculate_bounding_box(landmarks, image_width, image_height):
    """Calculate person's bounding box"""
    x_coordinates = []
    y_coordinates = []
    
    for landmark in landmarks:
        if landmark.visibility > 0.5:
            x_coordinates.append(landmark.x * image_width)
            y_coordinates.append(landmark.y * image_height)

    try:
        left_eye_y = landmarks[2].y * image_height
        right_eye_y = landmarks[5].y * image_height
        avg_eye_y = (left_eye_y + right_eye_y) / 2
        nose_y = landmarks[0].y * image_height
        head_top_offset = abs(nose_y - avg_eye_y) * 4
        y_coordinates.append(avg_eye_y - head_top_offset)
    except (IndexError, AttributeError):
        pass

    x_min, x_max = int(min(x_coordinates)), int(max(x_coordinates))
    y_min, y_max = int(min(y_coordinates)), int(max(y_coordinates))
    return (x_min, y_min, x_max, y_max), (y_max - y_min)

def generate_frames():
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while height_data["is_active"]:
            with camera_lock:
                if not camera.isOpened():
                    break
                success, frame = camera.read()
            
            if not success:
                break
            
            # A4 detection
            a4_box, a4_px_height = detect_a4_paper(frame)
            
            # Person detection
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            height_text = "Height: N/A"
            if results.pose_landmarks:
                person_box, person_px_height = calculate_bounding_box(
                    results.pose_landmarks.landmark, 
                    frame.shape[1], 
                    frame.shape[0]
                )
                
                if a4_px_height and person_px_height > 0:
                    scale = A4_REAL_HEIGHT / a4_px_height
                    real_height = person_px_height * scale
                    height_data["height"] = f"{real_height:.2f}m"
                    height_text = f"Height: {real_height:.2f}m"
                    
                    cv2.rectangle(frame, a4_box[:2], a4_box[2:], (0, 0, 255), 2)
                
                cv2.rectangle(frame, person_box[:2], person_box[2:], (0, 255, 0), 2)
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.putText(frame, height_text, (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start_system():
    global camera
    if not height_data["is_active"]:
        with camera_lock:
            if camera is None or not camera.isOpened():
                camera = cv2.VideoCapture(0)
        height_data["is_active"] = True
    return jsonify({"status": "System started", "height": height_data["height"]})

@app.route('/stop', methods=['POST'])
def stop_system():
    global camera
    height_data["is_active"] = False
    height_data["height"] = "N/A"
    with camera_lock:
        if camera and camera.isOpened():
            camera.release()
            camera = None
    return jsonify({"status": "System stopped"})

@app.route('/get_height')
def get_height():
    return jsonify(height_data)

if __name__ == '__main__':
    app.run(debug=True)