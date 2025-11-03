import cv2
import mediapipe as mp
import numpy as np
from core.utils import calculate_angle, log_reps, cleanup

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Global variables
count = 0
position = "down"
smoothed_fill_ratio = 0  # For smooth bar animation


# ------------------------------
# 🔹 Visualizer (inside this file)
# ------------------------------
def draw_concentration_bar(image, angle):
    """Draws a smooth progress bar based on arm curl angle (EMA smoothed)."""
    global smoothed_fill_ratio

    # Bar position and size
    bar_x, bar_y, bar_height, bar_width = 50, 100, 200, 30

    # Normalize angle → fill ratio (150° → 0%, 30° → 100%)
    target_fill_ratio = max(0, min(1, (150 - angle) / 120))

    # Apply Exponential Moving Average (EMA)
    alpha = 0.25
    smoothed_fill_ratio = alpha * target_fill_ratio + (1 - alpha) * smoothed_fill_ratio

    fill_height = int(smoothed_fill_ratio * bar_height)

    # Color based on curl depth
    if angle > 140:
        color = (0, 0, 255)  # Red (not curled)
    elif 70 <= angle <= 140:
        color = (0, 255, 255)  # Yellow (decent)
    else:
        color = (0, 255, 0)  # Green (perfect curl)

    # Draw background + fill
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), 2)
    cv2.rectangle(image, (bar_x, bar_y + (bar_height - fill_height)),
                  (bar_x + bar_width, bar_y + bar_height), color, -1)

    # Labels
    cv2.putText(image, "Curl", (bar_x - 10, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    cv2.putText(image, "Low", (bar_x + 40, bar_y + bar_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 1)
    cv2.putText(image, "High", (bar_x + 40, bar_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 1)


# ------------------------------
# 🔹 Frame Processing Logic
# ------------------------------
def process_frame(frame):
    """
    Processes a single frame:
    - Detects pose landmarks
    - Calculates arm angles
    - Counts bicep curls
    - Displays smooth progress + feedback
    """


    global count, position

    image = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    try:
                
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get keypoints for both arms
            r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate angles
            r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
            avg_angle = (r_angle + l_angle) / 2

            # Curl counting logic
            if r_angle > 140 and l_angle > 140:
                position = "down"
            if r_angle < 50 and l_angle < 50 and position == "down":
                position = "up"
                count += 1
                print(f"✅ Curl Counted! Total: {count}")

            # Feedback
            if avg_angle > 140:
                feedback = "Extend your arms!"
            elif avg_angle < 50:
                feedback = "Full curl!"
            else:
                feedback = "Good form!"

            # Draw progress bar + feedback text
            draw_concentration_bar(img_bgr, avg_angle)
            cv2.putText(img_bgr, f"Reps: {count}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img_bgr, feedback, (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw pose landmarks
            mp_drawing.draw_landmarks(img_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    except Exception as e:
        print(f"Error: {e}")

    return img_bgr, count
