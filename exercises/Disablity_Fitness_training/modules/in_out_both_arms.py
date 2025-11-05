# import cv2
# import mediapipe as mp
# import numpy as np
# import datetime
# import csv
# import streamlit as st
# import os
# import time

# # ---------- Streamlit Setup ----------
# st.set_page_config(page_title="In & Out Both Arms Tracker", layout="wide")
# st.title("💪 In & Out Both Arms Tracker")
# st.markdown("**Press 'Stop' in the sidebar or press 'Q' on your keyboard to end the session.**")

# # Sidebar Controls
# st.sidebar.header("🎥 Controls")
# run_camera = st.sidebar.checkbox("Start Camera", value=False)

# # ---------- Optional Reference Video ----------
# LOCAL_VIDEO_PATH = r"videos\in_out_reference.mp4"

# col1, col2 = st.columns(2)

# # ---------- CSV Logging ----------
# def log_reps(exercise_name, count):
#     filename = "exercise_log.csv"
#     try:
#         with open(filename, 'x', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(["Date", "Time", "Exercise", "Reps"])
#     except FileExistsError:
#         pass

#     with open(filename, 'a', newline='') as f:
#         writer = csv.writer(f)
#         now = datetime.datetime.now()
#         writer.writerow([now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), exercise_name, count])

# # ---------- Draw Horizontal Progress Bar ----------
# def draw_horizontal_bar(image, progress, position, label):
#     x, y = position
#     bar_width, bar_height = 300, 30
#     progress = np.clip(progress, 0, 1)

#     # Outline
#     cv2.rectangle(image, (x, y), (x + bar_width, y + bar_height), (50, 50, 50), 2)

#     # Fill
#     filled_width = int(bar_width * progress)
#     color = (0, int(255 * progress), 255 - int(255 * progress))
#     cv2.rectangle(image, (x, y), (x + filled_width, y + bar_height), color, -1)

#     # Label
#     cv2.putText(image, f"{label}: {int(progress * 100)}%", (x, y - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# # ---------- Calculate angle between 3 points ----------
# def calculate_angle(a, b, c):
#     a, b, c = np.array(a), np.array(b), np.array(c)
#     radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
#     angle = np.abs(radians * 180.0 / np.pi)
#     if angle > 180:
#         angle = 360 - angle
#     return angle

# # ---------- MediaPipe Setup ----------
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(min_detection_confidence=0.6,
#                     min_tracking_confidence=0.6)
# mp_drawing = mp.solutions.drawing_utils

# # ---------- Camera + Video Stream ----------
# if run_camera:
#     cap = cv2.VideoCapture(0)
#     ref_cap = cv2.VideoCapture(LOCAL_VIDEO_PATH) if os.path.exists(LOCAL_VIDEO_PATH) else None
#     count, direction = 0, 0  # 0 = extending, 1 = curling in
#     frame_placeholder = col1.empty()
#     ref_placeholder = col2.empty()

#     if not os.path.exists(LOCAL_VIDEO_PATH):
#         col2.warning(f"⚠️ Reference video not found: {LOCAL_VIDEO_PATH}")
#     else:
#         st.sidebar.success("✅ Reference video loaded successfully")

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             st.error("⚠️ Camera not found or cannot open!")
#             break

#         frame = cv2.flip(frame, 1)
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(rgb_frame)

#         if results.pose_landmarks:
#             try:
#                 lm = results.pose_landmarks.landmark

#                 # --- Right Arm ---
#                 r_shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
#                               lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
#                 r_elbow = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
#                            lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
#                 r_wrist = [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
#                            lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

#                 # --- Left Arm ---
#                 l_shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
#                               lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
#                 l_elbow = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
#                            lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
#                 l_wrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
#                            lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

#                 # Compute average elbow angle
#                 r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
#                 l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
#                 avg_angle = (r_angle + l_angle) / 2

#                 # Normalize progress
#                 progress = np.interp(avg_angle, [50, 170], [1, 0])

#                 # Draw progress bar
#                 draw_horizontal_bar(frame, progress, (50, 100), "Arm Curl Progress")

#                 # Rep counting
#                 if progress > 0.85:
#                     direction = 1
#                 if progress < 0.15 and direction == 1:
#                     count += 1
#                     direction = 0

#                 # Display info
#                 cv2.putText(frame, f"Reps: {count}", (30, 60),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#                 cv2.putText(frame, f"Avg Angle: {int(avg_angle)}°", (30, 150),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#                 # Draw pose landmarks
#                 mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#             except Exception as e:
#                 st.warning(f"Error in detection: {e}")

#         # Display live feed
#         frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

#         # Display reference video
#         if ref_cap and ref_cap.isOpened():
#             ref_ret, ref_frame = ref_cap.read()
#             if ref_ret:
#                 ref_frame = cv2.resize(ref_frame, (640, 480))
#                 ref_placeholder.image(cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB), channels="RGB")
#             else:
#                 ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#         time.sleep(0.03)

#     cap.release()
#     if ref_cap:
#         ref_cap.release()
#     cv2.destroyAllWindows()
#     log_reps("In & Out Both Arms", count)
#     st.success(f"✅ Session completed! Logged {count} reps.")

import cv2
import mediapipe as mp
import numpy as np
import time
import os
import streamlit as st

# -------------------------------------------------------------
# Shared Helpers (should exist globally in app.py)
# -------------------------------------------------------------
def log_reps(exercise_name, count, filename="exercise_log.csv"):
    import csv, datetime
    try:
        with open(filename, 'x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Date", "Time", "Exercise", "Reps"])
    except FileExistsError:
        pass
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        now = datetime.datetime.now()
        writer.writerow([now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), exercise_name, count])


def draw_horizontal_bar(image, progress, position, label):
    x, y = position
    bar_width, bar_height = 300, 30
    progress = np.clip(progress, 0, 1)
    cv2.rectangle(image, (x, y), (x + bar_width, y + bar_height), (50, 50, 50), 2)
    filled_width = int(bar_width * progress)
    color = (0, int(255 * progress), 255 - int(255 * progress))
    cv2.rectangle(image, (x, y), (x + filled_width, y + bar_height), color, -1)
    cv2.putText(image, f"{label}: {int(progress * 100)}%", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def calculate_angle(a, b, c):
    """Calculate the joint angle between three points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle


# -------------------------------------------------------------
# In & Out Both Arms Tracker Function
# -------------------------------------------------------------
def run_in_out_both_arms(video_path="videos/in_out_reference.mp4", log_file="exercise_log.csv"):
    """
    Tracks bicep curl movement (In & Out Both Arms) using elbow angles.
    """

    # Streamlit Layout
    st.subheader("💪 In & Out Both Arms Tracker")
    col1, col2 = st.columns(2)
    run_camera = st.checkbox("🎥 Start Camera", value=False)
    st.markdown("Press **Q** to stop anytime.")

    if not run_camera:
        st.info("Enable the camera to start tracking your arm curls.")
        return

    # MediaPipe setup
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
    mp_drawing = mp.solutions.drawing_utils

    # Camera setup
    cap = cv2.VideoCapture(0)
    ref_cap = cv2.VideoCapture(video_path) if os.path.exists(video_path) else None

    count, direction = 0, 0  # 0 = extending, 1 = curling in
    frame_placeholder = col1.empty()
    ref_placeholder = col2.empty()

    if ref_cap:
        st.sidebar.success("✅ Reference video loaded successfully.")
    else:
        st.sidebar.warning("⚠️ Reference video not found.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("⚠️ Camera not found or cannot open!")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            try:
                # --- Right Arm ---
                r_shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                r_elbow = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                r_wrist = [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # --- Left Arm ---
                l_shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                              lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                l_elbow = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                           lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                l_wrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                           lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Compute average elbow angle
                r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
                l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                avg_angle = (r_angle + l_angle) / 2

                # Normalize progress (170° = extended, 50° = curled)
                progress = np.interp(avg_angle, [50, 170], [1, 0])

                # Draw progress bar
                draw_horizontal_bar(frame, progress, (50, 100), "Arm Curl Progress")

                # Rep counting
                if progress > 0.85:
                    direction = 1
                if progress < 0.15 and direction == 1:
                    count += 1
                    direction = 0

                # Display information
                cv2.putText(frame, f"Reps: {count}", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Avg Angle: {int(avg_angle)}°", (30, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Draw pose landmarks
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            except Exception as e:
                st.warning(f"⚠️ Error in detection: {e}")

        # Show live feed
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # Show reference video
        if ref_cap and ref_cap.isOpened():
            ref_ret, ref_frame = ref_cap.read()
            if ref_ret:
                ref_frame = cv2.resize(ref_frame, (640, 480))
                ref_placeholder.image(cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB), channels="RGB")
            else:
                ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.03)

    # Cleanup
    cap.release()
    if ref_cap:
        ref_cap.release()
    cv2.destroyAllWindows()

    # Log & notify
    log_reps("In & Out Both Arms", count, log_file)
    st.success(f"✅ Session completed! Logged {count} reps.")
    return count






















