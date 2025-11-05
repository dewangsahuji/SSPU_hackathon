# import cv2
# import mediapipe as mp
# import numpy as np
# import datetime
# import csv
# import streamlit as st
# import os
# import time

# # ---------- Streamlit Setup ----------
# st.set_page_config(page_title="Adapted Burpees Tracker", layout="wide")
# st.title("🏋️ Adapted Burpees Tracker")
# st.markdown("**Press 'Stop' in the sidebar or press 'Q' on your keyboard to end the session.**")

# # Sidebar Controls
# st.sidebar.header("🎥 Controls")
# run_camera = st.sidebar.checkbox("Start Camera", value=False)

# # ---------- Fixed Local Video Path ----------
# LOCAL_VIDEO_PATH = r"videos\burpees_reference.mp4"

# # Layout Columns
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


# # ---------- Drawing Helper ----------
# def draw_progress_bar(image, progress, position, label):
#     x, y = position
#     bar_width, bar_height = 30, 300
#     filled_height = int((progress / 100) * bar_height)
#     cv2.rectangle(image, (x, y - bar_height), (x + bar_width, y), (50, 50, 50), 2)
#     color = (0, int(progress * 2.55), 255 - int(progress * 2.55))
#     cv2.rectangle(image, (x, y - filled_height), (x + bar_width, y), color, -1)
#     cv2.putText(image, label, (x - 10, y + 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#     cv2.putText(image, f"{int(progress)}%", (x - 10, y - bar_height - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


# # ---------- MediaPipe Setup ----------
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(min_detection_confidence=0.5,
#                     min_tracking_confidence=0.5)
# mp_drawing = mp.solutions.drawing_utils

# # ---------- Camera + Video Stream ----------
# if run_camera:
#     cap = cv2.VideoCapture(0)
#     ref_cap = cv2.VideoCapture(LOCAL_VIDEO_PATH) if os.path.exists(LOCAL_VIDEO_PATH) else None
#     count, direction = 0, 0  # direction = 0 (down), 1 (up)
#     frame_placeholder = col1.empty()
#     ref_placeholder = col2.empty()

#     if not os.path.exists(LOCAL_VIDEO_PATH):
#         col2.error(f"⚠️ Reference video not found: {LOCAL_VIDEO_PATH}")
#     else:
#         st.sidebar.success("✅ Reference video loaded successfully")

#     st.sidebar.write("Press **Q** to stop.")

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             st.error("⚠️ Camera not found or cannot open!")
#             break

#         frame = cv2.flip(frame, 1)
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(rgb_frame)

#         if results.pose_landmarks:
#             landmarks = results.pose_landmarks.landmark
#             try:
#                 # Get key landmarks
#                 r_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
#                 l_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
#                 r_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
#                 l_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
#                 nose_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y

#                 # Average positions
#                 wrist_y = (r_wrist_y + l_wrist_y) / 2
#                 hip_y = (r_hip_y + l_hip_y) / 2

#                 # Normalize progress: wrist near hip = 0, wrist near head = 1
#                 total_range = hip_y - nose_y
#                 progress = (hip_y - wrist_y) / total_range
#                 progress = np.clip(progress, 0, 1)
#                 progress_percent = progress * 100

#                 # Draw progress bar
#                 draw_progress_bar(frame, progress_percent, (80, 400), "Arm Raise")

#                 # Rep counting logic
#                 if progress > 0.8:
#                     direction = 1  # arms up
#                 if progress < 0.3 and direction == 1:
#                     count += 1
#                     direction = 0

#                 # Display count
#                 cv2.putText(frame, f"Burpees: {count}", (30, 50),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#                 # Draw skeleton
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
#                 ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#         time.sleep(0.03)  # reduce CPU load

#     cap.release()
#     if ref_cap:
#         ref_cap.release()
#     cv2.destroyAllWindows()
#     log_reps("Adapted Burpees", count)
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


def draw_progress_bar(image, progress, position, label):
    x, y = position
    bar_width, bar_height = 30, 300
    filled_height = int((progress / 100) * bar_height)
    cv2.rectangle(image, (x, y - bar_height), (x + bar_width, y), (50, 50, 50), 2)
    color = (0, int(progress * 2.55), 255 - int(progress * 2.55))
    cv2.rectangle(image, (x, y - filled_height), (x + bar_width, y), color, -1)
    cv2.putText(image, label, (x - 10, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(image, f"{int(progress)}%", (x - 10, y - bar_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


# -------------------------------------------------------------
# Adapted Burpees Tracker Function
# -------------------------------------------------------------
def run_adapted_burpees(video_path="videos/burpees_reference.mp4", log_file="exercise_log.csv"):
    """
    Runs Adapted Burpees tracker: counts arm raise reps using pose detection.
    """

    # Streamlit Layout
    st.subheader("🏋️ Adapted Burpees Tracker")
    col1, col2 = st.columns(2)
    run_camera = st.checkbox("🎥 Start Camera", value=False)
    st.markdown("Press **Q** to stop anytime.")

    if not run_camera:
        st.info("Enable the camera to begin your Adapted Burpees session.")
        return

    # MediaPipe setup
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Camera setup
    cap = cv2.VideoCapture(0)
    ref_cap = cv2.VideoCapture(video_path) if os.path.exists(video_path) else None

    count, direction = 0, 0  # 0 = down, 1 = up
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

        # Flip & process
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            try:
                # Extract key landmarks
                r_wrist_y = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
                l_wrist_y = lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y
                r_hip_y = lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y
                l_hip_y = lm[mp_pose.PoseLandmark.LEFT_HIP.value].y
                nose_y = lm[mp_pose.PoseLandmark.NOSE.value].y

                # Compute averages
                wrist_y = (r_wrist_y + l_wrist_y) / 2
                hip_y = (r_hip_y + l_hip_y) / 2

                # Normalize movement progress (0 = down, 1 = arms up)
                total_range = hip_y - nose_y
                progress = (hip_y - wrist_y) / total_range
                progress = np.clip(progress, 0, 1)
                progress_percent = progress * 100

                # Draw progress
                draw_progress_bar(frame, progress_percent, (80, 400), "Arm Raise")

                # Rep counting logic
                if progress > 0.8:
                    direction = 1
                if progress < 0.3 and direction == 1:
                    count += 1
                    direction = 0

                # Display info
                cv2.putText(frame, f"Burpees: {count}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Draw landmarks
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            except Exception as e:
                st.warning(f"⚠️ Error during detection: {e}")

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

        time.sleep(0.03)  # prevent CPU overload

    # Cleanup
    cap.release()
    if ref_cap:
        ref_cap.release()
    cv2.destroyAllWindows()

    # Log and display
    log_reps("Adapted Burpees", count, log_file)
    st.success(f"✅ Session completed! Logged {count} reps.")
    return count






























