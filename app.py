import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt

st.title("Real-time Human Pose Detection and Saving Joint Coordinates")

# 버튼 설정
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    start_button = st.button('Start')
with col2:
    finish_button = st.button('Finish')
with col3:
    evaluate_button = st.button('Evaluate')
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
    return csv_filename

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle

# Start 버튼이 눌렸을 때
if start_button:
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    pose_data = []

    cap = cv2.VideoCapture(0)
    
    # 비디오 저장 설정 (Mac과 Windows 모두에서 열릴 수 있도록)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v'는 더 많은 브라우저와 플레이어에서 호환됩니다.
    video_filename = os.path.join(video_path, f'output_{current_time}.mp4')
    out = cv2.VideoWriter(video_filename, fourcc, 15.0, (640, 480))

    knee_angle_threshold = 90  # 목표 무릎 각도
    knee_angle_frame = None

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
        
        # 실시간으로 CSV 저장
        save_pose_data(pose_data, current_time)
        
        # 비디오에 프레임 추가
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        
        if finish_button:
            break
    
    cap.release()
    out.release()
    save_pose_data(pose_data, current_time)

    cap = None
    out = None

# Finish 버튼이 눌렸을 때
if finish_button and cap is not None:
    cap.release()
    out.release()
    save_pose_data(pose_data, current_time)
    cap = None
    out = None

# Evaluate 버튼이 눌렸을 때
if evaluate_button:
    # 가장 최근에 생성된 CSV 파일과 비디오 파일 찾기
    csv_files = sorted([f for f in os.listdir(csv_path) if f.endswith('.csv')])
    video_files = sorted([f for f in os.listdir(video_path) if f.endswith('.mp4')])
    
    if csv_files and video_files:
        latest_csv = csv_files[-1]
        latest_video = video_files[-1]
        
        # CSV 데이터 로드
        df = pd.read_csv(os.path.join(csv_path, latest_csv))
        
        # 각도 계산을 위한 주요 랜드마크 추출 함수
        def get_landmarks(df, index):
            return np.column_stack((df[f"x_{index}"].values, df[f"y_{index}"].values))

        hip = get_landmarks(df, mp_pose.PoseLandmark.LEFT_HIP.value)
        knee = get_landmarks(df, mp_pose.PoseLandmark.LEFT_KNEE.value)
        ankle = get_landmarks(df, mp_pose.PoseLandmark.LEFT_ANKLE.value)
        shoulder = get_landmarks(df, mp_pose.PoseLandmark.LEFT_SHOULDER.value)

        # 각도와 속도 계산
        knee_angle = np.array([calculate_angle(hip[i], knee[i], ankle[i]) for i in range(len(df))])
        hip_angle = np.array([calculate_angle(shoulder[i], hip[i], knee[i]) for i in range(len(df))])
        ankle_angle = np.array([calculate_angle(knee[i], ankle[i], [ankle[i][0], ankle[i][1] + 0.1]) for i in range(len(df))])
        
        knee_y = df[f"y_{mp_pose.PoseLandmark.LEFT_KNEE.value}"].values
        squat_speed = np.diff(knee_y)  # y 좌표 변화로 속도 계산

        # 각도와 속도 시각화
        fig, ax = plt.subplots(5, 1, figsize=(10, 15))
        fig.subplots_adjust(hspace=1.0)

        ax[0].plot(knee_angle, label='Knee Angle')
        ax[0].set_title('Knee Angle Over Time')
        ax[0].set_xlabel('Frame')
        ax[0].set_ylabel('Angle')
        ax[0].legend()
        ax[0].grid(True)
        
        ax[1].plot(hip_angle, label='Hip Angle', color='orange')
        ax[1].set_title('Hip Angle Over Time')
        ax[1].set_xlabel('Frame')
        ax[1].set_ylabel('Angle')
        ax[1].legend()
        ax[1].grid(True)
        
        ax[2].plot(ankle_angle, label='Ankle Angle', color='green')
        ax[2].set_title('Ankle Angle Over Time')
        ax[2].set_xlabel('Frame')
        ax[2].set_ylabel('Angle')
        ax[2].legend()
        ax[2].grid(True)
        
        ax[3].plot(squat_speed, label='Squat Speed', color='red')
        ax[3].set_title('Squat Speed Over Time')
        ax[3].set_xlabel('Frame')
        ax[3].set_ylabel('Speed')
        ax[3].legend()
        ax[3].grid(True)
        
        ax[4].plot(knee_y, label='Knee Y Coordinate', color='blue')
        ax[4].set_title('Knee Y Coordinate Over Time')
        ax[4].set_xlabel('Frame')
        ax[4].set_ylabel('Y Coordinate')
        ax[4].legend()
        ax[4].grid(True)

        st.pyplot(fig)
        
        st.write(f"CSV File: {latest_csv}")
        st.write(f"Video File: {latest_video}")

        # 비디오 재생
        video_file_path = os.path.join(video_path, latest_video)
        st.video(video_file_path)
    else:
        st.write("No files found for evaluation.")