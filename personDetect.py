import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

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

    x_min = int(min(x_coordinates))
    x_max = int(max(x_coordinates))
    y_min = int(min(y_coordinates))
    y_max = int(max(y_coordinates))
    return (x_min, y_min, x_max, y_max), (y_max - y_min)

def main():
    cap = cv2.VideoCapture(0)
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            # Detect person
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            if results.pose_landmarks:
                # Calculate person bounding box
                person_box, person_height = calculate_bounding_box(
                    results.pose_landmarks.landmark, 
                    frame.shape[1], 
                    frame.shape[0]
                )
                
                # Draw elements
                cv2.rectangle(frame, (person_box[0], person_box[1]), 
                            (person_box[2], person_box[3]), (0, 255, 0), 2)
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Display height
                text = f"Pixel Height: {person_height}"
                cv2.putText(frame, text, (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Person Detection', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
