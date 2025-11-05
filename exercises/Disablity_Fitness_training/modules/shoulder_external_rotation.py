# import cv2
# import mediapipe as mp
# import numpy as np
# import datetime
# import csv
# import streamlit as st
# import os
# import time

# # ---------- Streamlit Setup ----------
# st.set_page_config(page_title="Shoulder External Rotation Tracker", layout="wide")
# st.title("🏋️ Shoulder External Rotation Tracker")
# st.markdown("**Press 'Stop' in the sidebar or press 'Q' on your keyboard to end the session.**")

# # Sidebar Controls
# st.sidebar.header("🎥 Controls")
# run_camera = st.sidebar.checkbox("Start Camera", value=False)

# # ---------- Fixed local video path ----------
# LOCAL_VIDEO_PATH = r"videos\1.mp4"

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
#     bar_width, bar_height = 30, 200
#     filled_height = int((progress / 100) * bar_height)
#     cv2.rectangle(image, (x, y - bar_height), (x + bar_width, y), (50, 50, 50), 2)
#     color = (0, int(progress * 2.55), 255 - int(progress * 2.55))
#     cv2.rectangle(image, (x, y - filled_height), (x + bar_width, y), color, -1)
#     cv2.putText(image, label, (x - 20, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#     cv2.putText(image, f"{int(progress)}%", (x - 10, y - bar_height - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# # ---------- MediaPipe Setup ----------
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# mp_drawing = mp.solutions.drawing_utils

# # ---------- Camera + Video Stream ----------
# if run_camera:
#     cap = cv2.VideoCapture(0)
#     ref_cap = cv2.VideoCapture(LOCAL_VIDEO_PATH) if os.path.exists(LOCAL_VIDEO_PATH) else None
#     count, direction = 0, 0
#     frame_placeholder = col1.empty()
#     ref_placeholder = col2.empty()

#     if not os.path.exists(LOCAL_VIDEO_PATH):
#         col2.error(f"⚠️ Video not found at: {LOCAL_VIDEO_PATH}")
#     else:
#         st.sidebar.success("✅ Reference video loaded successfully")

#     st.sidebar.write("Press **Q** to stop.")

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             st.error("Camera not found!")
#             break

#         frame = cv2.flip(frame, 1)
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(rgb_frame)

#         if results.pose_landmarks:
#             landmarks = results.pose_landmarks.landmark
#             try:
#                 r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
#                 r_hip_opposite = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
#                 l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
#                 l_hip_opposite = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

#                 r_dist = np.sqrt((r_wrist.x - r_hip_opposite.x)**2 + (r_wrist.y - r_hip_opposite.y)**2)
#                 l_dist = np.sqrt((l_wrist.x - l_hip_opposite.x)**2 + (l_wrist.y - l_hip_opposite.y)**2)

#                 r_progress = np.interp(r_dist, [0.15, 0.35], [0, 100])
#                 l_progress = np.interp(l_dist, [0.15, 0.35], [0, 100])
#                 r_progress = np.clip(r_progress, 0, 100)
#                 l_progress = np.clip(l_progress, 0, 100)

#                 draw_progress_bar(frame, r_progress, (80, 400), "Right")
#                 draw_progress_bar(frame, l_progress, (140, 400), "Left")

#                 if r_progress > 80 or l_progress > 80:
#                     direction = 1
#                 if (r_progress < 30 or l_progress < 30) and direction == 1:
#                     count += 1
#                     direction = 0

#                 cv2.putText(frame, f"Reps: {count}", (30, 50),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#                 mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#             except Exception as e:
#                 st.warning(f"Error in detection: {e}")

#         # Show camera frame
#         frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

#         # Show reference video
#         if ref_cap and ref_cap.isOpened():
#             ref_ret, ref_frame = ref_cap.read()
#             if ref_ret:
#                 ref_frame = cv2.resize(ref_frame, (640, 480))
#                 ref_placeholder.image(cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB), channels="RGB")
#             else:
#                 ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#         time.sleep(0.03)  # prevent CPU overload

#     cap.release()
#     if ref_cap:
#         ref_cap.release()
#     cv2.destroyAllWindows()
#     log_reps("Shoulder External Rotation (Distance)", count)
#     st.success(f"✅ Session completed! Logged {count} reps.")












import cv2
import mediapipe as mp
import numpy as np
import time
import os
import streamlit as st

# -------------------------------------------------------------
# Helper functions (imported or defined globally in app.py)
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
    bar_width, bar_height = 30, 200
    filled_height = int((progress / 100) * bar_height)
    cv2.rectangle(image, (x, y - bar_height), (x + bar_width, y), (50, 50, 50), 2)
    color = (0, int(progress * 2.55), 255 - int(progress * 2.55))
    cv2.rectangle(image, (x, y - filled_height), (x + bar_width, y), color, -1)
    cv2.putText(image, label, (x - 20, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(image, f"{int(progress)}%", (x - 10, y - bar_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


# -------------------------------------------------------------
# Shoulder External Rotation Tracker Function
# -------------------------------------------------------------
def run_shoulder_external_rotation(video_path="videos/External_rotation.mp4", log_file="exercise_log.csv"):
    """
    Runs Shoulder External Rotation exercise tracker with webcam input
    and optional reference video playback.
    """

    # Streamlit layout
    st.subheader("🏋️ Shoulder External Rotation Tracker")
    col1, col2 = st.columns(2)

    # Sidebar control
    run_camera = st.checkbox("🎥 Start Camera", value=False)
    st.markdown("Press **Q** to stop at any time.")

    if not run_camera:
        st.info("Turn on the camera from the checkbox above to start tracking.")
        return

    # MediaPipe setup
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Camera setup
    cap = cv2.VideoCapture(0)
    ref_cap = cv2.VideoCapture(video_path) if os.path.exists(video_path) else None

    count, direction = 0, 0
    frame_placeholder = col1.empty()
    ref_placeholder = col2.empty()

    if ref_cap:
        st.sidebar.success("✅ Reference video loaded successfully.")
    else:
        st.sidebar.warning("⚠️ No reference video found.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("⚠️ Camera not detected or unavailable.")
            break

        # Flip & process
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            try:
                lm = results.pose_landmarks.landmark
                # Extract key points
                r_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                l_wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST.value]
                r_hip_opposite = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
                l_hip_opposite = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]

                # Calculate distances
                r_dist = np.sqrt((r_wrist.x - r_hip_opposite.x) ** 2 + (r_wrist.y - r_hip_opposite.y) ** 2)
                l_dist = np.sqrt((l_wrist.x - l_hip_opposite.x) ** 2 + (l_wrist.y - l_hip_opposite.y) ** 2)

                # Map to progress (0–100)
                r_progress = np.clip(np.interp(r_dist, [0.15, 0.35], [0, 100]), 0, 100)
                l_progress = np.clip(np.interp(l_dist, [0.15, 0.35], [0, 100]), 0, 100)

                # Draw progress bars
                draw_progress_bar(frame, r_progress, (80, 400), "Right")
                draw_progress_bar(frame, l_progress, (140, 400), "Left")

                # Rep counting
                if r_progress > 80 or l_progress > 80:
                    direction = 1
                if (r_progress < 30 or l_progress < 30) and direction == 1:
                    count += 1
                    direction = 0

                # Draw info & skeleton
                cv2.putText(frame, f"Reps: {count}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            except Exception as e:
                st.warning(f"⚠️ Error in detection: {e}")

        # Streamlit display
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
    log_reps("Shoulder External Rotation", count, log_file)
    st.success(f"✅ Session completed! Logged {count} reps.")
    return count

