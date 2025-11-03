import cv2
import mediapipe as mp
import numpy as np
import time
from core.utils import calculate_angle, log_reps

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Global variables
count = 0
position = "up"
smoothed_fill_ratio = 0
last_feedback = "Start your squats"
last_feedback_time = time.time()
feedback_delay = 1.5  # seconds between feedback changes


# ------------------------------
# 🔹 Visualizer (inside this file)
# ------------------------------
def draw_concentration_bar(image, angle):
    """
    Draws a smooth squat depth bar based on the knee angle.
    """
    global smoothed_fill_ratio

    bar_x, bar_y, bar_height, bar_width = 50, 100, 200, 30

    # Normalize angle to fill ratio (160° → 0%, 80° → 100%)
    target_fill_ratio = max(0, min(1, (160 - angle) / 80))

    # Smooth bar animation
    alpha = 0.25
    smoothed_fill_ratio = alpha * target_fill_ratio + (1 - alpha) * smoothed_fill_ratio
    fill_height = int(smoothed_fill_ratio * bar_height)

    # Depth color logic
    if angle > 140:
        color = (0, 0, 255)  # Red: Not deep enough
    elif 90 <= angle <= 140:
        color = (0, 255, 255)  # Yellow: Okay depth
    else:
        color = (0, 255, 0)  # Green: Perfect depth

    # Draw bar background and fill
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), 2)
    cv2.rectangle(image, (bar_x, bar_y + (bar_height - fill_height)),
                  (bar_x + bar_width, bar_y + bar_height), color, -1)

    # Text labels
    cv2.putText(image, "Depth", (bar_x - 10, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(image, "Low", (bar_x + 40, bar_y + bar_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(image, "High", (bar_x + 40, bar_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


# ------------------------------
# 🔹 Frame Processing Logic
# ------------------------------
def process_frame(frame):
    """
    Processes a frame for squat detection:
    - Detects pose
    - Calculates hip/knee/ankle angles
    - Counts reps
    - Provides feedback
    - Draws visualizer and landmarks
    """
    global count, position, last_feedback, last_feedback_time

    image = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    try:
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Key body points
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

            # Angle calculation
            angle = calculate_angle(hip, knee, ankle)

            # Squat counting logic
            if angle > 160:
                position = "up"
            if angle < 90 and position == "up":
                position = "down"
                count += 1
                print(f"✅ Squat Counted! Total: {count}")

            # Feedback (limited frequency)
            new_feedback = last_feedback
            if angle > 160:
                new_feedback = "Stand tall!"
            elif angle < 90:
                new_feedback = "Squat low!"
            else:
                new_feedback = "Good depth!"

            if time.time() - last_feedback_time > feedback_delay:
                last_feedback = new_feedback
                last_feedback_time = time.time()

            # Visualizer + Text
            draw_concentration_bar(img_bgr, angle)
            cv2.putText(img_bgr, f"Squat Reps: {count}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img_bgr, last_feedback, (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw landmarks
            mp_drawing.draw_landmarks(img_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    except Exception as e:
        print(f"Error: {e}")

    return img_bgr, count
