import cv2
import numpy as np

# Constants for A4 detection
A4_ASPECT_RATIO = 210/297  # Width/Height ratio
ASPECT_TOLERANCE = 0.15

def detect_a4_paper(frame):
    """Detect A4 paper using contour analysis"""
    # Convert to HSV and threshold for white paper
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 50, 255])
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
                return (x, y, x+w, y+h), (w, h)
    return None, None

def main():
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Detect A4 paper
        a4_box, a4_dimensions = detect_a4_paper(frame)
        
        if a4_box:
            # Draw A4 paper bounding box
            cv2.rectangle(frame, (a4_box[0], a4_box[1]), 
                        (a4_box[2], a4_box[3]), (0, 0, 255), 2)
            
            # Display dimensions
            text = f"A4 Size: {a4_dimensions[0]}x{a4_dimensions[1]}px"
            cv2.putText(frame, text, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('A4 Paper Detection', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
