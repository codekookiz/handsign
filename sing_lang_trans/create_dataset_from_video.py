import cv2
import sys
import os
import mediapipe as mp
import numpy as np
import modules.holistic_module as hm
from modules.utils import createDirectory, Vector_Normalization
import time

# 출력 비디오를 저장할 디렉토리 생성
createDirectory('dataset/output_video')

# 설정
seq_length = 10  # 시퀀스 길이
actions = ['아프다', '열', "기침", "콧물", "코막힘", "머리", "배", "설사", "변비", "구토",
    "메스껍다", "어지럽다", "피곤하다", "숨차다", "가슴답답", "땀난다", "감기", "몸살", "요통", "근육통",
    "쑤시다", "통증", "따갑다", "저리다", "가렵다", "붓다", "딱딱하다", "눌리다", "움직이기 힘들다", "눕기 어렵다",
    "먹기 힘들다", "삼키기 어렵다", "소화 안된다", "체했다", "열이 내리지 않는다", "지속된다", "가슴통증", "호흡곤란", "귀가 아프다", "목이 아프다",
    "침 삼키기 어렵다", "말하기 힘들다", "소리 안 들린다", "시야 흐리다", "눈이 아프다", "눈물난다", "귀에서 소리난다", "속쓰리다", "명치 아프다", "설태 있다",
    "배가 더부룩하다", "배가 나왔다", "오한이 있다", "손 떨린다", "다리 저리다", "허리 아프다", "허리 삐끗했다", "다쳤다", "출혈 있다", "멍들었다",
    "부딪혔다", "넘어졌다", "피난다", "살짝 긁혔다", "베였다", "찔렸다", "화상 입었다", "타박상", "골절 같다", "뼈 부러졌다",
    "탈골된 것 같다", "관절 아프다", "무릎 아프다", "발목 아프다", "팔꿈치 아프다", "어깨 결린다", "뻐근하다", "쥐났다", "경련 있다", "근육 당긴다",
    "입안 헐었다", "잇몸 부었다", "이 아프다", "혀가 아프다", "목마르다", "갈증 난다", "토할 것 같다", "체한 것 같다", "열 식었다", "몸이 나른하다",
    "잠이 안 온다", "불안하다", "긴장된다", "불편하다", "답답하다", "가슴이 쿵쾅거린다", "식은땀 난다", "기운 없다", "기절할 것 같다", "쓰러질 것 같다"
]  # 인식할 동작 목록

# 데이터셋 초기화
dataset = {i: [] for i in range(len(actions))}

# MediaPipe 홀리스틱 모델 초기화
detector = hm.HolisticDetector(min_detection_confidence=0.3)

# 비디오 파일 목록 생성
videoFolderPath = "dataset/output_video"
testTargetList = []

def get_video_files(folder_path):
    """
    지정된 폴더에서 모든 비디오 파일의 경로를 재귀적으로 찾아 리스트로 반환
    """
    for videoPath in os.listdir(folder_path):
        actionVideoPath = os.path.join(folder_path, videoPath)
        for actionVideo in os.listdir(actionVideoPath):
            fullVideoPath = os.path.join(actionVideoPath, actionVideo)
            testTargetList.append(fullVideoPath)

get_video_files(videoFolderPath)

# 비디오 파일 목록을 정렬 (파일명의 두 번째 부분을 기준으로 역순 정렬)
testTargetList = sorted(testTargetList, key=lambda x: x.split("/")[-2], reverse=True)
print("Video List:", testTargetList)

def process_video(video_path):
    """
    주어진 비디오 파일을 처리하여 핸드 랜드마크 데이터를 추출
    """
    data = []
    idx = actions.index(video_path.split("/")[-2])
    print("Now Streaming:", video_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Camera open failed!")
        sys.exit()

    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = round(1000/fps) if fps != 0 else round(1000/30)

    while True:
        ret, img = cap.read()
        if not ret:
            break

        img = detector.findHolistic(img, draw=True)
        _, right_hand_lmList = detector.findRighthandLandmark(img)

        if right_hand_lmList is not None:
            joint = np.zeros((42, 2))
            for j, lm in enumerate(right_hand_lmList.landmark):
                joint[j] = [lm.x, lm.y]

            # 벡터 정규화 및 각도 계산
            vector, angle_label = Vector_Normalization(joint)
            angle_label = np.append(angle_label, idx)
            d = np.concatenate([vector.flatten(), angle_label.flatten()])
            data.append(d)

        cv2.waitKey(delay)
        if cv2.waitKey(delay) == 27:  # ESC 키를 누르면 종료
            break

    print("Finish Video Streaming")
    return np.array(data), idx

# 각 비디오 처리
for target in testTargetList:
    data, idx = process_video(target)
    
    # 시퀀스 데이터 생성
    for seq in range(len(data) - seq_length):
        dataset[idx].append(data[seq:seq + seq_length])

# 데이터 저장 (현재 주석 처리됨)

def save_dataset():
    """
    추출된 데이터셋을 파일로 저장
    """
    created_time = int(time.time())
    for i in range(len(actions)):
        save_data = np.array(dataset[i])
        np.save(os.path.join('dataset', f'seq_{actions[i]}_{created_time}'), save_data)
    print("Finish Save Dataset")

save_dataset()