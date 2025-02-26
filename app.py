
import streamlit as st
import cv2
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from models.yolo_model import YOLOv10Model
from tracking.deepsort_tracker import DeepSORTTracker
from utils.visualization import draw_boxes
from utils.processing import read_video
import os
import time

# Capture timestamp for each frame
timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

# List to store blind spot analysis data
blind_spot_data = []
frame_number = 0  # Initialize frame counter

# Define the save path for the CSV file
csv_save_path = "blind_spot_analysis_log.csv"

st.set_page_config(page_title="Blind Spot Analysis", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f5f7fa;
        }
        .main-title {
            font-size: 28px;
            font-weight: bold;
            color: #2c3e50;
            text-align: left;
            padding-bottom: 10px;
        }
        .description {
            font-size: 16px;
            color: #555;
            text-align: justify;
            line-height: 1.6;
            background: #ffffff;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            background-color: #3498db;
            color: white;
            border: none;
            height: 40px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #2980b9;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown('<div class="main-title">Blind Spot Detection & Analysis</div>', unsafe_allow_html=True)

# Project Description
st.markdown(
    """
   <div class="description">
        This project focuses on <b>real-time object tracking and detection</b> to enhance road safety by identifying vehicles moving towards a two-wheeler’s <b>blind spot</b>.
        Utilizing <b>YOLOv10</b> for detection and <b>DeepSORT</b> for tracking, the system processes <b>images, videos, and live webcam feeds</b> to provide accurate vehicle movement insights.
        This project is ideal for enhancing <b>situational awareness, reducing blind spot accidents, and assisting in intelligent traffic monitoring</b>.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# **📷 Input, Model, and Confidence - Inline UI**
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    input_option = st.selectbox("Select Input Source Type", ["Video", "Live Webcam"])

with col2:
    model_option = st.selectbox("YOLO Model Selection", ["yolov10n", "yolov10s", "yolov10m"])

with col3:
    confidence = st.slider("Confidence Threshold", 0, 100, 30, help="Higher confidence reduces false detections.")

# **📁 File Upload (for Video/Image only)**
uploaded_file = st.file_uploader("📂 Upload Image/Video", type=["jpg", "png", "mp4"])

st.markdown("---")

# **⚙ Load YOLOv10 and DeepSORT**
model = YOLOv10Model(model_option)
tracker = DeepSORTTracker()

col1, col2 = st.columns([1, 1])

if input_option == "Video" and uploaded_file:
    video_path = read_video(uploaded_file)
    cap = cv2.VideoCapture(video_path)
    stframe1, stframe2 = col1.empty(), col2.empty()
    img_height, img_width = None, None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if img_height is None or img_width is None:
            img_height, img_width, _ = frame.shape

        detections = model.detect_objects(frame, confidence / 100)
        tracked_objects = tracker.track_objects(detections, frame)
        frame = draw_boxes(frame, tracked_objects, model.model.names)

        with col1:
            stframe1.image(frame, channels="BGR")

        with col2:
            fig = go.Figure()

            # Define blind spot center based on bike's rear camera position
            blind_spot_x = img_width / 2  # Centered
            blind_spot_y = img_height * 0.9  # Lower position to match rear wheel
            radius = img_width * 0.1  # Slightly larger zone for better coverage

            # Define the camera FOV angle (adjust as needed)
            fov_angle = np.pi / 1.5  # 120 degrees (wider for bike's rear view)

            # Generate points for the arc (rear view FOV simulation)
            theta = np.linspace(-fov_angle / 2, fov_angle / 2, 30)  # Wider field of view
            arc_x = blind_spot_x + radius * np.sin(theta)  # X-coordinates (flip for rear view)
            arc_y = blind_spot_y - radius * np.cos(theta)  # Y-coordinates (inverted for rear)

            # Invert y-coordinates to match OpenCV convention
            arc_y = [img_height - y for y in arc_y]

            # **Plot Rear Camera Blind Spot**
            fig.add_trace(go.Scatter(
                x=np.append(arc_x, blind_spot_x),  # Close the arc
                y=np.append(arc_y, img_height - blind_spot_y),  # Close the arc
                fill='toself',
                fillcolor='rgba(255, 0, 0, 0.3)',  # Light red transparent fill
                line=dict(color='rgba(255, 0, 0, 0.5)'),
                name="Blind Spot Zone"
            ))

            # **Plot Bike Rear Camera Position (Red Dot)**
            fig.add_trace(go.Scatter(
                x=[blind_spot_x],
                y=[img_height - blind_spot_y],  # Adjust for OpenCV coordinate system
                mode="markers",
                marker=dict(color="red", size=10),
                name="Two-Wheeler"
            ))

            # Plot tracked objects dynamically
            for track in tracked_objects:
                if track.is_confirmed() and track.time_since_update == 0:
                    x, y, w, h = map(int, track.to_tlwh())  # Bounding box
                    track_id = track.track_id
                    obj_class = model.model.names.get(track.det_class, "Unknown")

                    # Calculate distance (approximation using bounding box height)
                    focal_length = 700  # You need to calibrate this value
                    known_object_height = 1.5  # Example: Average car height in meters
                    distance = (known_object_height * focal_length) / h if h > 0 else -1  # Avoid division by zero

                    # Check if object is inside the blind spot
                    in_blind_spot = (blind_spot_x - radius <= x + w / 2 <= blind_spot_x + radius) and \
                                    (blind_spot_y - radius <= y + h / 2 <= blind_spot_y + radius)

                    # Store data in list
                    blind_spot_data.append([timestamp, track_id, obj_class, x, y, w, h, distance, in_blind_spot])

                    # Ensure bounding box is within frame bounds
                    x = max(0, min(x, img_width - 1))
                    y = max(0, min(y, img_height - 1))
                    w = max(1, min(w, img_width - x))
                    h = max(1, min(h, img_height - y))

                    # Define color mapping for detected objects
                    object_colors = {
                        "car": "green",
                        "truck": "blue",
                        "motorcycle": "red",
                        "bus": "yellow"
                    }

                    # Get the predefined color for the object type or default to gray
                    object_color = object_colors.get(obj_class, "gray")

                    # Add a circle marker for the detected object
                    fig.add_trace(go.Scatter(
                        x=[x + w / 2],
                        y=[img_height - (y + h / 2)],
                        mode="markers",
                        marker=dict(color=object_color, size=10),
                        name=f"ID {track_id} ({obj_class})"
                    ))

            fig.update_layout(
                title="Blind Spot Analysis",
                xaxis=dict(title="X-axis"),
                yaxis=dict(title="Y-axis"),
                showlegend=True
            )

            # Update the graph in real-time
            stframe2.plotly_chart(fig, use_container_width=True, key=f"plot_{int(cv2.getTickCount())}")

    cap.release()
    os.remove(video_path)

    # Save analysis data to CSV after processing the video
    if blind_spot_data:
        df = pd.DataFrame(blind_spot_data,
                          columns=["Timestamp", "Track ID", "Class", "X", "Y", "Width", "Height", "Distance (m)",
                                   "In Blind Spot"])

        df.to_csv(csv_save_path, index=False)

        st.success(f"Blind Spot Analysis Log has been saved to {os.path.abspath(csv_save_path)}")
        st.download_button(
            label="📥 Download Analysis Log",
            data=open(csv_save_path, "rb"),
            file_name="blind_spot_analysis_log.csv",
            mime="text/csv"
        )


elif input_option == "Live Webcam":
    cap = cv2.VideoCapture(0)
    stframe1, stframe2 = col1.empty(), col2.empty()
    img_height, img_width = None, None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if img_height is None or img_width is None:
            img_height, img_width, _ = frame.shape

        detections = model.detect_objects(frame, confidence / 100)
        tracked_objects = tracker.track_objects(detections, frame)
        frame = draw_boxes(frame, tracked_objects, model.model.names)

        with col1:
            stframe1.image(frame, channels="BGR")

        with col2:
            fig = go.Figure()

            # Define blind spot center based on bike's rear camera position
            blind_spot_x = img_width / 2  # Centered
            blind_spot_y = img_height * 0.9  # Lower position to match rear wheel
            radius = img_width * 0.1  # Slightly larger zone for better coverage

            # Define the camera FOV angle (adjust as needed)
            fov_angle = np.pi / 1.5  # 120 degrees (wider for bike's rear view)

            # Generate points for the arc (rear view FOV simulation)
            theta = np.linspace(-fov_angle / 2, fov_angle / 2, 30)  # Wider field of view
            arc_x = blind_spot_x + radius * np.sin(theta)  # X-coordinates (flip for rear view)
            arc_y = blind_spot_y - radius * np.cos(theta)  # Y-coordinates (inverted for rear)

            # Invert y-coordinates to match OpenCV convention
            arc_y = [img_height - y for y in arc_y]

            # **Plot Rear Camera Blind Spot**
            fig.add_trace(go.Scatter(
                x=np.append(arc_x, blind_spot_x),  # Close the arc
                y=np.append(arc_y, img_height - blind_spot_y),  # Close the arc
                fill='toself',
                fillcolor='rgba(255, 0, 0, 0.3)',  # Light red transparent fill
                line=dict(color='rgba(255, 0, 0, 0.5)'),
                name="Blind Spot Zone"
            ))

            # **Plot Bike Rear Camera Position (Red Dot)**
            fig.add_trace(go.Scatter(
                x=[blind_spot_x],
                y=[img_height - blind_spot_y],  # Adjust for OpenCV coordinate system
                mode="markers",
                marker=dict(color="red", size=10),
                name="Two-Wheeler"
            ))

            # Plot tracked objects dynamically
            for track in tracked_objects:
                if track.is_confirmed() and track.time_since_update == 0:
                    x, y, w, h = map(int, track.to_tlwh())  # Bounding box
                    track_id = track.track_id
                    obj_class = model.model.names.get(track.det_class, "Unknown")

                    # Calculate distance (approximation using bounding box height)
                    focal_length = 700  # You need to calibrate this value
                    known_object_height = 1.5  # Example: Average car height in meters
                    distance = (known_object_height * focal_length) / h if h > 0 else -1  # Avoid division by zero

                    # Check if object is inside the blind spot
                    in_blind_spot = (blind_spot_x - radius <= x + w / 2 <= blind_spot_x + radius) and \
                                    (blind_spot_y - radius <= y + h / 2 <= blind_spot_y + radius)

                    # Store data in list
                    blind_spot_data.append([timestamp, track_id, obj_class, x, y, w, h, distance, in_blind_spot])

                    # Ensure bounding box is within frame bounds
                    x = max(0, min(x, img_width - 1))
                    y = max(0, min(y, img_height - 1))
                    w = max(1, min(w, img_width - x))
                    h = max(1, min(h, img_height - y))

                    # Define color mapping for detected objects
                    object_colors = {
                        "car": "green",
                        "truck": "blue",
                        "motorcycle": "red",
                        "bus": "yellow"
                    }

                    # Get the predefined color for the object type or default to gray
                    object_color = object_colors.get(obj_class, "gray")

                    # Add a circle marker for the detected object
                    fig.add_trace(go.Scatter(
                        x=[x + w / 2],
                        y=[img_height - (y + h / 2)],
                        mode="markers",
                        marker=dict(color=object_color, size=10),
                        name=f"ID {track_id} ({obj_class})"
                    ))

            fig.update_layout(
                title="Blind Spot Analysis",
                xaxis=dict(title="X-axis"),
                yaxis=dict(title="Y-axis"),
                showlegend=True
            )

            # Update the graph in real-time
            stframe2.plotly_chart(fig, use_container_width=True, key=f"plot_{int(cv2.getTickCount())}")

    cap.release()  # Stop capturing from webcam

    # Save analysis data to CSV after processing the live stream
    if blind_spot_data:
        df = pd.DataFrame(blind_spot_data,
                          columns=["Timestamp", "Track ID", "Class", "X", "Y", "Width", "Height", "Distance (m)",
                                   "In Blind Spot"])

        df.to_csv(csv_save_path, index=False)

        st.success(f"Blind Spot Analysis Log has been saved to {os.path.abspath(csv_save_path)}")
        st.download_button(
            label="📥 Download Analysis Log",
            data=open(csv_save_path, "rb"),
            file_name="blind_spot_analysis_log.csv",
            mime="text/csv"
        )





