# import cv2
# import mediapipe as mp
# import numpy as np
# import datetime
# import csv
# import streamlit as st
# import os
# import time

# # ---------- Streamlit Setup ----------
# st.set_page_config(page_title="Side Reach (Both Arms) Tracker", layout="wide")
# st.title("💪 Side Reach (Both Arms) Tracker")
# st.markdown("**Press 'Stop' in the sidebar or press 'Q' on your keyboard to end the session.**")

# # Sidebar Controls
# st.sidebar.header("🎥 Controls")
# run_camera = st.sidebar.checkbox("Start Camera", value=False)

# # ---------- Fixed Local Video Path ----------
# LOCAL_VIDEO_PATH = r"videos\side_reach_reference.mp4"

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

#     total_count = 0
#     right_count, left_count = 0, 0
#     right_dir, left_dir = 0, 0  # 0 = down, 1 = up

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
#                 lm = results.pose_landmarks.landmark

#                 # --- Right Arm ---
#                 r_hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
#                          lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
#                 r_wrist = [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
#                            lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

#                 # --- Left Arm ---
#                 l_hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x,
#                          lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
#                 l_wrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
#                            lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

#                 # Calculate vertical distances (hip to wrist)
#                 r_dist = r_hip[1] - r_wrist[1]
#                 l_dist = l_hip[1] - l_wrist[1]

#                 # Normalize progress (tuned for sensitivity)
#                 r_progress = np.interp(r_dist, [0.05, 0.45], [0, 1])
#                 l_progress = np.interp(l_dist, [0.05, 0.45], [0, 1])

#                 # Draw progress bars
#                 draw_horizontal_bar(frame, r_progress, (50, 100), "Right Hand Reach")
#                 draw_horizontal_bar(frame, l_progress, (50, 150), "Left Hand Reach")

#                 # ----- RIGHT ARM -----
#                 if r_progress > 0.85:
#                     right_dir = 1
#                 if r_progress < 0.15 and right_dir == 1:
#                     right_count += 1
#                     total_count += 1
#                     right_dir = 0

#                 # ----- LEFT ARM -----
#                 if l_progress > 0.85:
#                     left_dir = 1
#                 if l_progress < 0.15 and left_dir == 1:
#                     left_count += 1
#                     total_count += 1
#                     left_dir = 0

#                 # Display counts
#                 cv2.putText(frame, f"Total Reps: {total_count}", (30, 50),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#                 cv2.putText(frame, f"Right: {right_count}", (400, 120),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#                 cv2.putText(frame, f"Left: {left_count}", (400, 170),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#                 # Draw Pose Landmarks
#                 mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#             except Exception as e:
#                 st.warning(f"Error: {e}")

#         # Display live camera
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

#     log_reps("Side Reach (Both Arms)", total_count)
#     st.success(f"✅ Session completed! Logged {total_count} total reps.")



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


# -------------------------------------------------------------
# Side Reach (Both Arms) Tracker Function
# -------------------------------------------------------------
def run_side_reach(video_path="videos/side_reach_reference.mp4", log_file="exercise_log.csv"):
    """
    Tracks side reach movement for both arms using MediaPipe Pose.
    """

    # Streamlit Layout
    st.subheader("💪 Side Reach (Both Arms) Tracker")
    col1, col2 = st.columns(2)
    run_camera = st.checkbox("🎥 Start Camera", value=False)
    st.markdown("Press **Q** to stop anytime.")

    if not run_camera:
        st.info("Enable the camera to begin your side reach workout.")
        return

    # MediaPipe setup
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
    mp_drawing = mp.solutions.drawing_utils

    # Camera setup
    cap = cv2.VideoCapture(0)
    ref_cap = cv2.VideoCapture(video_path) if os.path.exists(video_path) else None

    total_count, right_count, left_count = 0, 0, 0
    right_dir, left_dir = 0, 0  # 0 = down, 1 = up
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
                # --- Right Arm ---
                r_hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                r_wrist = [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # --- Left Arm ---
                l_hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                         lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                l_wrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                           lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Calculate vertical distances (hip to wrist)
                r_dist = r_hip[1] - r_wrist[1]
                l_dist = l_hip[1] - l_wrist[1]

                # Normalize progress
                r_progress = np.interp(r_dist, [0.05, 0.45], [0, 1])
                l_progress = np.interp(l_dist, [0.05, 0.45], [0, 1])

                # Draw progress bars
                draw_horizontal_bar(frame, r_progress, (50, 100), "Right Hand Reach")
                draw_horizontal_bar(frame, l_progress, (50, 150), "Left Hand Reach")

                # ----- RIGHT ARM -----
                if r_progress > 0.85:
                    right_dir = 1
                if r_progress < 0.15 and right_dir == 1:
                    right_count += 1
                    total_count += 1
                    right_dir = 0

                # ----- LEFT ARM -----
                if l_progress > 0.85:
                    left_dir = 1
                if l_progress < 0.15 and left_dir == 1:
                    left_count += 1
                    total_count += 1
                    left_dir = 0

                # Display counts
                cv2.putText(frame, f"Total Reps: {total_count}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Right: {right_count}", (400, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Left: {left_count}", (400, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Draw Pose Landmarks
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            except Exception as e:
                st.warning(f"⚠️ Error during detection: {e}")

        # Display live camera
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

        time.sleep(0.03)

    # Cleanup
    cap.release()
    if ref_cap:
        ref_cap.release()
    cv2.destroyAllWindows()

    # Log and notify
    log_reps("Side Reach (Both Arms)", total_count, log_file)
    st.success(f"✅ Session completed! Logged {total_count} total reps.")
    return total_count













