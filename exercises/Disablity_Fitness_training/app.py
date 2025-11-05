import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# Import your modular trackers
from modules.shoulder_external_rotation import run_shoulder_external_rotation
from modules.adapted_burpees import run_adapted_burpees
from modules.seated_wood_chop import run_seated_wood_chop
from modules.in_out_both_arms import run_in_out_both_arms
from modules.side_reach import run_side_reach

# -------------------------------------------------------------
# Streamlit Setup
# -------------------------------------------------------------
st.set_page_config(page_title="Fitoproto Exercise Tracker", layout="wide")
st.title("🏋️ Fitoproto — AI Exercise Tracker")
st.markdown("Track your workouts, visualize progress, and improve your form with real-time AI feedback.")

# -------------------------------------------------------------
# Tab Layout
# -------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🏋️ Exercise Tracker", "📊 Progress Dashboard", "🎥 Reference Videos"])

# -------------------------------------------------------------
# 🏋️ TAB 1: Exercise Tracker
# -------------------------------------------------------------
with tab1:
    st.header("Select an Exercise to Begin")

    exercise = st.selectbox(
        "Choose your workout:",
        [
            "Shoulder External Rotation",
            "Adapted Burpees",
            "Seated Wood Chop",
            "In & Out Both Arms",
            "Side Reach (Both Arms)"
        ],
        index=0
    )

    st.markdown("---")

    if exercise == "Shoulder External Rotation":
        run_shoulder_external_rotation()
    elif exercise == "Adapted Burpees":
        run_adapted_burpees()
    elif exercise == "Seated Wood Chop":
        run_seated_wood_chop()
    elif exercise == "In & Out Both Arms":
        run_in_out_both_arms()
    elif exercise == "Side Reach (Both Arms)":
        run_side_reach()

# -------------------------------------------------------------
# 📊 TAB 2: Progress Dashboard
# -------------------------------------------------------------
with tab2:
    st.header("Progress Dashboard")

    log_path = "exercise_log.csv"
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        st.dataframe(df)

        if not df.empty:
            st.subheader("📈 Total Reps per Exercise")
            summary = df.groupby("Exercise")["Reps"].sum().reset_index()

            plt.figure(figsize=(8, 4))
            plt.bar(summary["Exercise"], summary["Reps"])
            plt.title("Total Reps by Exercise")
            plt.xticks(rotation=30)
            plt.ylabel("Reps")
            st.pyplot(plt)

            st.subheader("🕒 Recent Sessions")
            st.dataframe(df.tail(10))
        else:
            st.info("No exercise data available yet. Perform a session to see progress.")
    else:
        st.warning("⚠️ Log file not found. Start a session to create one.")

# -------------------------------------------------------------
# 🎥 TAB 3: Reference Videos
# -------------------------------------------------------------
with tab3:
    st.header("Reference Videos")

    videos = {
        "Shoulder External Rotation": "videos/External_rotation.mp4",
        "Adapted Burpees": "videos/burpees_reference.mp4",
        "Seated Wood Chop": "videos/wood_chop_reference.mp4",
        "In & Out Both Arms": "videos/in_out_reference.mp4",
        "Side Reach (Both Arms)": "videos/side_reach_reference.mp4"
    }

    for name, path in videos.items():
        st.markdown(f"### 🎬 {name}")
        if os.path.exists(path):
            st.video(path)
        else:
            st.warning(f"⚠️ Missing reference video: `{path}`")
