import cv2
import mediapipe as mp
import numpy as np
import csv
import datetime
import time



# ------------------------------
# 🔹 Log Exercise Reps to CSV
# ------------------------------
# Function to log exercise reps in CSV

def log_reps(exercise_name, count):
    filename = "exercise_log.csv"
    
    # Check if the file exists, if not, add headers
    try:
        with open(filename, 'x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Date", "Time", "Exercise", "Reps"])
    except FileExistsError:
        pass  # File already exists, no need to write headers
    
    # Append new data
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d,%H:%M:%S")
        writer.writerow([timestamp, exercise_name, count])
    
    print(f"✅ Saved {count} reps of {exercise_name} to exercise_log.csv")



# -----------------------------------
# to close the program much better 
# ------------------------------------
def cleanup():
    cap.release()
    cv2.destroyAllWindows()
    exit()  # Immediate program termination


# ------------------------------
# 🔹 Calculate Joint Angle
# ------------------------------
# Function to calculate the angle between three points (for joint angle calculations)

def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint (joint)
    c = np.array(c)  # End point
    
    # Calculate the angle using arctan2 function
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    # Adjust angle to stay within 0-180 degrees
    if angle > 180.0:
        angle = 360 - angle
    
    return angle