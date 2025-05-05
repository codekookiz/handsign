import cv2
import numpy as np
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
from tensorflow.keras.models import load_model

# 설정
actions = [
    "답답하다", "땀난다", "베였다", "부었다", "불편하다",
    "아프다", "어지럽다", "열난다", "지속된다", "피곤하다"
]
seq_length = 10
USED_FACE_INDEXES = [
    33, 133, 362, 263, 1, 61, 291, 199, 429,
    152, 234, 454, 138, 172, 136, 215, 177, 152  # 18개
]

# 폰트 로드
try:
    font = ImageFont.truetype("fonts/HMKMMAG.TTF", 40)
except:
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/AppleGothic.ttf", 40)

def draw_text_on_image(img, text):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    w, h = img_pil.size
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    position = ((w - text_w) / 2, h - text_h - 30)

    margin = 10
    draw.rectangle(
        [position[0] - margin, position[1] - margin,
         position[0] + text_w + margin, position[1] + text_h + margin],
        fill=(0, 0, 0, 180)
    )
    draw.text(position, text, font=font, fill=(255, 255, 255))
    return np.array(img_pil)

def main():
    model = load_model('models/facial_expression_symptom_classifier.h5')
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(0)

    seq = []
    action_seq = []
    last_action = None

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]

            coords = []
            for i in USED_FACE_INDEXES:
                try:
                    lm = face.landmark[i]
                    coords.append([lm.x, lm.y, lm.z])
                except:
                    coords.append([0.0, 0.0, 0.0])  # 누락 대비

            coords = np.array(coords).flatten()
            if coords.shape[0] != 56:
                print(f"[SKIP] 피처 개수 불일치: {coords.shape[0]}")
                continue

            seq.append(coords)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
            y_pred = model.predict(input_data)[0]
            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            if conf < 0.9 or i_pred >= len(actions):
                continue

            action = actions[i_pred]
            action_seq.append(action)

            if len(action_seq) >= 3 and action_seq[-1] == action_seq[-2] == action_seq[-3]:
                if last_action != action:
                    last_action = action
                    img = draw_text_on_image(img, f'{action}')

        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()