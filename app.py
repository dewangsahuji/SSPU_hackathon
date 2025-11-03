from flask import Flask, render_template, Response
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import io, base64
from flask import request, jsonify


#-------------------------------
# Import your exercise trackers
#-------------------------------

from exercises.bicep.bicep_tracker import process_frame as bicep_frame
from exercises.squat.squat_tracker import process_frame as squat_frame
from exercises.pushup.pushup_tracker import process_frame as pushup_frame

# Initialize Flask app
app = Flask(__name__)

# Initialize webcam
camera = cv2.VideoCapture(0)

# Frame generator function
def generate_frames(exercise):
    """
    Captures frames from the webcam and processes them
    according to the selected exercise.
    """
    frame_func = {
        'bicep': bicep_frame,
        'squat': squat_frame,
        'pushup': pushup_frame
    }.get(exercise)

    if not frame_func:
        raise ValueError(f"Unknown exercise: {exercise}")

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Process each frame with the respective exercise logic
            frame, count = frame_func(frame)

            # Encode the frame for MJPEG streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame to the web browser
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            

# Flask Routes
@app.route('/')
def index():
    """Home page where user selects an exercise"""
    return render_template('index.html')

@app.route('/start/<exercise>')
def start_exercise(exercise):
    """Exercise tracking page"""
    return render_template('exercise.html', exercise=exercise)

@app.route('/video_feed/<exercise>')
def video_feed(exercise):
    """Live video stream route"""
    return Response(generate_frames(exercise),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ------------------------------
# 🔹 Dashboard Route
# ------------------------------
@app.route('/dashboard')
def dashboard():
    try:
        # Read exercise log
        df = pd.read_csv('exercise_log.csv')

        # If no data yet
        if df.empty:
            return render_template('dashboard.html', chart=None, message="No exercise data found yet!")

        # Group total reps per exercise
        summary = df.groupby('Exercise')['Reps'].sum().sort_values(ascending=False)

        # Create bar chart
        fig, ax = plt.subplots(figsize=(6, 4))
        summary.plot(kind='bar', ax=ax)
        ax.set_title('Total Reps per Exercise')
        ax.set_ylabel('Reps')
        ax.set_xlabel('Exercise')
        ax.grid(True, linestyle='--', alpha=0.5)

        # Save plot to base64 string
        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format='png')
        buf.seek(0)
        encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        plt.close(fig)

        # Render dashboard with chart
        return render_template('dashboard.html', chart=encoded, message=None)

    except FileNotFoundError:
        return render_template('dashboard.html', chart=None, message="No data file found! Please do an exercise first.")



# ------------------------------
# 🔹 Fitobot Chatbot Routes
# ------------------------------

# ✅ Chatbot routes
@app.route('/chatbot', methods=['GET'])
def chatbot():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    reply = get_fitobot_response(user_input)
    return jsonify({"response": reply})



















if __name__ == "__main__":
    app.run(debug=True)
