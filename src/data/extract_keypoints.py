"""
Trich xuat pose keypoints bang MediaPipe Tasks API (PoseLandmarker + HandLandmarker)
Da toi uu toc do: dung ban Pose Lite + resize khung hinh xuong 640px truoc khi xu ly
(khong anh huong do chinh xac landmark vi MediaPipe tra toa do da chuan hoa 0-1).

Chay: python -m src.data.extract_keypoints
"""
import csv
import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm

TRANSCRIPT_CSV = "data/raw/transcripts.csv"
OUT_DIR = "data/processed/keypoints"
MANIFEST_CSV = "data/processed/manifest.csv"

POSE_MODEL_PATH = "models/pose_landmarker_lite.task"  # doi tu _full sang _lite de tang toc
HAND_MODEL_PATH = "models/hand_landmarker.task"

MAX_FRAMES = 256
RESIZE_WIDTH = 640  # giam do phan giai truoc khi dua vao MediaPipe, tang toc dang ke


def build_landmarkers():
    pose_options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
    )
    pose_landmarker = mp_vision.PoseLandmarker.create_from_options(pose_options)
    hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)
    return pose_landmarker, hand_landmarker


def resize_frame(frame):
    h, w = frame.shape[:2]
    if w <= RESIZE_WIDTH:
        return frame
    scale = RESIZE_WIDTH / w
    new_h = int(h * scale)
    return cv2.resize(frame, (RESIZE_WIDTH, new_h), interpolation=cv2.INTER_AREA)


def landmarks_to_vec(landmark_list, n_points):
    if landmark_list is None:
        return np.zeros(n_points * 3)
    return np.array([[lm.x, lm.y, lm.z] for lm in landmark_list]).flatten()


def extract_frame_vec(pose_result, hand_result):
    pose_landmarks = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None
    pose_vec = landmarks_to_vec(pose_landmarks, 33)

    left_hand, right_hand = None, None
    for i, handedness in enumerate(hand_result.handedness):
        label = handedness[0].category_name
        if label == "Left":
            left_hand = hand_result.hand_landmarks[i]
        elif label == "Right":
            right_hand = hand_result.hand_landmarks[i]

    left_vec = landmarks_to_vec(left_hand, 21)
    right_vec = landmarks_to_vec(right_hand, 21)

    return np.concatenate([pose_vec, left_vec, right_vec])


def normalize(sequence: np.ndarray) -> np.ndarray:
    left_shoulder_idx = 11 * 3
    right_shoulder_idx = 12 * 3
    center = (sequence[:, left_shoulder_idx:left_shoulder_idx + 2] +
              sequence[:, right_shoulder_idx:right_shoulder_idx + 2]) / 2
    sequence = sequence.copy()
    for i in range(0, sequence.shape[1], 3):
        sequence[:, i:i + 2] -= center
    return sequence


def pad_or_truncate(sequence: np.ndarray, max_len: int) -> np.ndarray:
    T = sequence.shape[0]
    if T >= max_len:
        return sequence[:max_len]
    pad = np.zeros((max_len - T, sequence.shape[1]))
    return np.concatenate([sequence, pad], axis=0)


def process_video(video_path: str, pose_landmarker, hand_landmarker) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames_landmarks = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = resize_frame(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int((frame_idx / fps) * 1000)

        pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        frames_landmarks.append(extract_frame_vec(pose_result, hand_result))
        frame_idx += 1

    cap.release()
    sequence = np.array(frames_landmarks)
    if sequence.shape[0] == 0:
        return np.zeros((MAX_FRAMES, 225))
    sequence = normalize(sequence)
    sequence = pad_or_truncate(sequence, MAX_FRAMES)
    return sequence


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(TRANSCRIPT_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    pose_landmarker, hand_landmarker = build_landmarkers()

    manifest_rows = []
    for row in tqdm(rows, desc="Trich xuat keypoints"):
        idx = row["idx"]
        video_path = row["video_path"]
        text = row["text"]

        sequence = process_video(video_path, pose_landmarker, hand_landmarker)
        npy_path = os.path.join(OUT_DIR, f"{idx}.npy")
        np.save(npy_path, sequence)

        manifest_rows.append({"idx": idx, "npy_path": npy_path, "text": text})

    pose_landmarker.close()
    hand_landmarker.close()

    with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "npy_path", "text"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nHoan tat. Manifest luu tai {MANIFEST_CSV}")


if __name__ == "__main__":
    main()