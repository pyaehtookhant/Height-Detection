import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Constants
A4_REAL_HEIGHT = 0.297  # A4 height in meters (297mm)
A4_ASPECT_RATIO = 210/297  # Width/Height ratio
ASPECT_TOLERANCE = 0.15

def detect_a4_paper(frame):
    """Detect A4 paper using contour analysis"""
    # Convert to HSV and threshold for white paper
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 30, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        # Approximate contour
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
        
        if len(approx) == 4:
            # Calculate aspect ratio
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / h
            
            # Check aspect ratio with tolerance
            if abs(aspect - A4_ASPECT_RATIO) < ASPECT_TOLERANCE:
                return (x, y, x+w, y+h), h  # Return bounding box and height
    return None, None

def calculate_bounding_box(landmarks, image_width, image_height):
    """Calculate bounding box including estimated head top"""
    x_coordinates = []
    y_coordinates = []
    
    for landmark in landmarks:
        if landmark.visibility > 0.5:
            x_coordinates.append(landmark.x * image_width)
            y_coordinates.append(landmark.y * image_height)

    # Head top estimation
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

def main():
    cap = cv2.VideoCapture(0)
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            # Detect A4 paper
            a4_box, a4_px_height = detect_a4_paper(frame)
            
            # Detect person
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            height_text = "Height: N/A"
            if results.pose_landmarks:
                # Calculate person bounding box and pixel height
                person_box, person_px_height = calculate_bounding_box(
                    results.pose_landmarks.landmark, 
                    frame.shape[1], 
                    frame.shape[0]
                )
                
                if a4_px_height and person_px_height > 0:
                    # Calculate scale factor
                    scale = A4_REAL_HEIGHT / a4_px_height
                    real_height = person_px_height * scale
                    height_text = f"Height: {real_height:.2f}m"
                    
                    # Draw A4 paper
                    cv2.rectangle(frame, (a4_box[0], a4_box[1]), 
                                (a4_box[2], a4_box[3]), (0, 0, 255), 2)
                
                # Draw person bounding box and landmarks
                cv2.rectangle(frame, (person_box[0], person_box[1]), 
                              (person_box[2], person_box[3]), (0, 255, 0), 2)
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Display height
                cv2.putText(frame, height_text, (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Height Measurement System', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
