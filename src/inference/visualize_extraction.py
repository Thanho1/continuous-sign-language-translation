"""
Xuat 1 video co ve landmark (khung xuong) chong len video goc, de chung minh
truc quan buoc "doc frame -> trich xuat hanh dong" cho thay thay ro rang.

Chay: python -m src.inference.visualize_extraction --video test_video.mp4
"""
import argparse
import os

import cv2
import mediapipe as mp

from src.data.extract_keypoints import build_landmarkers

OUTPUT_DIR = "outputs"


def draw_landmarks_on_frame(frame, pose_result, hand_result):
    h, w = frame.shape[:2]

    # Ve pose (mau xanh la)
    if pose_result.pose_landmarks:
        for lm in pose_result.pose_landmarks[0]:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    # Ve 2 tay (mau do)
    for hand_landmarks in hand_result.hand_landmarks:
        for lm in hand_landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    return frame


def main(video_path: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "extraction_visualized.mp4")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    pose_landmarker, hand_landmarker = build_landmarkers()

    frame_idx = 0
    print("Dang xu ly va ve landmark tren tung frame...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int((frame_idx / fps) * 1000)

        pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        annotated = draw_landmarks_on_frame(frame.copy(), pose_result, hand_result)
        cv2.putText(annotated, f"Frame {frame_idx}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()
    pose_landmarker.close()
    hand_landmarker.close()

    print(f"\nHoan tat. Da xu ly {frame_idx} frame.")
    print(f"Video minh hoa (co landmark) luu tai: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    args = parser.parse_args()
    main(args.video)