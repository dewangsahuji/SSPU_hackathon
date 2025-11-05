import cv2
import mediapipe as mp
import numpy as np
from core.utils import calculate_angle, log_reps

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Global variables
count = 0
position = "up"
smoothed_fill_ratio = 0


# ------------------------------
# 🔹 Visualizer (inside this file)
# ------------------------------
def draw_concentration_bar(image, angle):
    """
    Draws a progress bar representing pushup depth (based on elbow angle).
    """
    global smoothed_fill_ratio

    bar_x, bar_y, bar_height, bar_width = 50, 100, 200, 30

    # Normalize angle → fill ratio (170° → 0%, 70° → 100%)
    target_fill_ratio = max(0, min(1, (170 - angle) / 100))

    # Smooth animation
    alpha = 0.25
    smoothed_fill_ratio = alpha * target_fill_ratio + (1 - alpha) * smoothed_fill_ratio
    fill_height = int(smoothed_fill_ratio * bar_height)

    # Color logic
    if angle > 160:
        color = (0, 0, 255)  # Red: Not low enough
    elif 90 <= angle <= 160:
        color = (0, 255, 255)  # Yellow: Medium depth
    else:
        color = (0, 255, 0)  # Green: Perfect pushup

    # Draw bar + fill
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), 2)
    cv2.rectangle(image, (bar_x, bar_y + (bar_height - fill_height)),
                  (bar_x + bar_width, bar_y + bar_height), color, -1)

    # Labels
    cv2.putText(image, "Depth", (bar_x - 10, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(image, "Low", (bar_x + 40, bar_y + bar_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(image, "High", (bar_x + 40, bar_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


# ------------------------------
# 🔹 Frame Processing Logic
# ------------------------------
def process_frame(frame):
    """
    Processes a frame for pushup detection:
    - Tracks elbow movement
    - Counts reps
    - Displays visual feedback and bar
    """
    global count, position

    image = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    try:
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Left side points
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate elbow angle
            angle = calculate_angle(shoulder, elbow, wrist)

            # Pushup counting logic
            if angle > 160:
                position = "up"
            if angle < 90 and position == "up":
                position = "down"
                count += 1
                print(f"✅ Pushup Counted! Total: {count}")

            # Feedback based on form
            if angle > 160:
                feedback = "Go lower!"
            elif angle < 90:
                feedback = "Nice depth!"
            else:
                feedback = "Good form!"

            # Draw visualizer
            draw_concentration_bar(img_bgr, angle)

            # Text overlay
            cv2.putText(img_bgr, f"Pushup Reps: {count}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img_bgr, feedback, (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw landmarks
            mp_drawing.draw_landmarks(img_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    except Exception as e:
        print(f"Error: {e}")

    return img_bgr, count

