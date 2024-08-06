import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from datetime import datetime

st.title("Real-time Human Pose Detection and Saving Joint Coordinates")

# 버튼 설정
col1, col2 = st.columns([1, 1])
with col1:
    run_button = st.button('Start')
with col2:
    stop_button = st.button('Stop')
FRAME_WINDOW = st.image([])

# Mediapipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 데이터 저장 경로 설정
base_path = '/Users/y3korea/Documents/pose_sample_test'
csv_path = os.path.join(base_path, 'csv')
video_path = os.path.join(base_path, 'video')
os.makedirs(csv_path, exist_ok=True)
os.makedirs(video_path, exist_ok=True)

pose_data = []
cap = None
out = None

def save_pose_data(pose_data, current_time):
    csv_filename = os.path.join(csv_path, f'pose_data_{current_time}.csv')
    column_names = []
    for i in range(33):
        column_names.extend([f"x_{i}", f"y_{i}", f"z_{i}", f"visibility_{i}"])
    pd.DataFrame(pose_data, columns=column_names).to_csv(csv_filename, index=False)

# Start 버튼이 눌렸을 때
if run_button:
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    pose_data = []

    cap = cv2.VideoCapture(0)
    
    # 비디오 저장 설정 (Mac과 Windows 모두에서 열릴 수 있도록)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v'는 Mac과 Windows 모두에서 호환됩니다.
    out = cv2.VideoWriter(os.path.join(video_path, f'output_{current_time}.mp4'), fourcc, 15.0, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame)
        
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            pose_row = list(np.array([[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in landmarks]).flatten())
            pose_data.append(pose_row)
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        FRAME_WINDOW.image(frame)
        
        # 비디오에 프레임 추가
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        
        if stop_button:
            break
    
    cap.release()
    out.release()
    save_pose_data(pose_data, current_time)
    cap = None
    out = None

# Stop 버튼이 눌렸을 때
if stop_button and cap is not None:
    cap.release()
    out.release()
    save_pose_data(pose_data, current_time)
    cap = None
    out = None