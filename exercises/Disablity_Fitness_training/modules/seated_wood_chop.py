# import cv2
# import mediapipe as mp
# import numpy as np
# import datetime
# import csv
# import streamlit as st
# import os
# import time

# # ---------- Streamlit Setup ----------
# st.set_page_config(page_title="Seated Wood Chop Tracker", layout="wide")
# st.title("🪓 Seated Wood Chop Tracker")
# st.markdown("**Press 'Stop' in the sidebar or press 'Q' on your keyboard to end the session.**")

# # Sidebar Controls
# st.sidebar.header("🎥 Controls")
# run_camera = st.sidebar.checkbox("Start Camera", value=False)

# # ---------- Fixed Local Video Path ----------
# LOCAL_VIDEO_PATH = r"videos\wood_chop_reference.mp4"

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


# # ---------- MediaPipe Setup ----------
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(min_detection_confidence=0.6,
#                     min_tracking_confidence=0.6)
# mp_drawing = mp.solutions.drawing_utils

# # ---------- Camera + Video Stream ----------
# if run_camera:
#     cap = cv2.VideoCapture(0)
#     ref_cap = cv2.VideoCapture(LOCAL_VIDEO_PATH) if os.path.exists(LOCAL_VIDEO_PATH) else None
#     count, direction = 0, 0  # 0 = down-right, 1 = up-left
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
#             try:
#                 landmarks = results.pose_landmarks.landmark

#                 # Use wrists and hips to detect diagonal swing
#                 r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
#                 l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
#                 r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
#                 l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]

#                 # Average wrist position
#                 wrist_x = (r_wrist.x + l_wrist.x) / 2
#                 wrist_y = (r_wrist.y + l_wrist.y) / 2

#                 # Define start (upper left near shoulder) and end (lower right near hip)
#                 start_x, start_y = l_shoulder.x, l_shoulder.y
#                 end_x, end_y = r_hip.x, r_hip.y

#                 # Compute normalized progress along diagonal
#                 dx = end_x - start_x
#                 dy = end_y - start_y
#                 vector_length = np.sqrt(dx**2 + dy**2)
#                 proj = ((wrist_x - start_x) * dx + (wrist_y - start_y) * dy) / (vector_length**2)
#                 progress = np.clip(proj, 0, 1)

#                 # Draw progress bar
#                 draw_horizontal_bar(frame, progress, (50, 100), "Diagonal Swing")

#                 # Rep counting
#                 if progress > 0.85:
#                     direction = 1
#                 if progress < 0.15 and direction == 1:
#                     count += 1
#                     direction = 0

#                 # Display count
#                 cv2.putText(frame, f"Wood Chops: {count}", (30, 60),
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
#                 ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#         time.sleep(0.03)  # reduce CPU load

#     cap.release()
#     if ref_cap:
#         ref_cap.release()
#     cv2.destroyAllWindows()
#     log_reps("Seated Wood Chop", count)
#     st.success(f"✅ Session completed! Logged {count} reps.")










import cv2
import mediapipe as mp
import numpy as np
import time
import os
import streamlit as st

# -------------------------------------------------------------
# Shared Helpers (should exist globally or be imported in app.py)
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


# -------------------------------------------------------------
# Seated Wood Chop Function
# -------------------------------------------------------------
def run_seated_wood_chop(video_path="videos/wood_chop_reference.mp4", log_file="exercise_log.csv"):
    """
    Runs Seated Wood Chop tracker: counts diagonal swing reps using MediaPipe Pose.
    """

    # Streamlit Layout
    st.subheader("🪓 Seated Wood Chop Tracker")
    col1, col2 = st.columns(2)
    run_camera = st.checkbox("🎥 Start Camera", value=False)
    st.markdown("Press **Q** to stop at any time.")

    if not run_camera:
        st.info("Turn on the camera to start tracking your movement.")
        return

    # MediaPipe setup
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
    mp_drawing = mp.solutions.drawing_utils

    # Camera setup
    cap = cv2.VideoCapture(0)
    ref_cap = cv2.VideoCapture(video_path) if os.path.exists(video_path) else None

    count, direction = 0, 0  # 0 = down-right, 1 = up-left
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
            try:
                lm = results.pose_landmarks.landmark

                # Use wrists and hips to detect diagonal swing
                r_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                l_wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST.value]
                r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
                l_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]

                # Average wrist position
                wrist_x = (r_wrist.x + l_wrist.x) / 2
                wrist_y = (r_wrist.y + l_wrist.y) / 2

                # Define start (upper-left near shoulder) and end (lower-right near hip)
                start_x, start_y = l_shoulder.x, l_shoulder.y
                end_x, end_y = r_hip.x, r_hip.y

                # Compute normalized progress along the diagonal
                dx = end_x - start_x
                dy = end_y - start_y
                vector_length = np.sqrt(dx**2 + dy**2)
                proj = ((wrist_x - start_x) * dx + (wrist_y - start_y) * dy) / (vector_length**2)
                progress = np.clip(proj, 0, 1)

                # Draw progress bar
                draw_horizontal_bar(frame, progress, (50, 100), "Diagonal Swing")

                # Rep counting
                if progress > 0.85:
                    direction = 1
                if progress < 0.15 and direction == 1:
                    count += 1
                    direction = 0

                # Display info
                cv2.putText(frame, f"Wood Chops: {count}", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Draw skeleton
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            except Exception as e:
                st.warning(f"⚠️ Error in detection: {e}")

        # Display live feed
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # Display reference video
        if ref_cap and ref_cap.isOpened():
            ref_ret, ref_frame = ref_cap.read()
            if ref_ret:
                ref_frame = cv2.resize(ref_frame, (640, 480))
                ref_placeholder.image(cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB), channels="RGB")
            else:
                ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.03)  # Reduce CPU load

    # Cleanup
    cap.release()
    if ref_cap:
        ref_cap.release()
    cv2.destroyAllWindows()

    # Log and notify
    log_reps("Seated Wood Chop", count, log_file)
    st.success(f"✅ Session completed! Logged {count} reps.")
    return count
