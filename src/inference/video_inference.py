"""
UC2: Dich tu 1 file video co san -> cau tieng Anh -> giong noi.
Da toi uu: Pose Lite model + resize 640px truoc khi xu ly MediaPipe.

Chay:
    python -m src.inference.video_inference --video path/to/video.mp4
"""

import argparse
import asyncio
import os
import time
from pathlib import Path

import cv2
import edge_tts
import mediapipe as mp
import numpy as np
import torch

from src.data.extract_keypoints import (
    build_landmarkers,
    extract_frame_vec,
    normalize,
    resize_frame,
)
from src.models.gloss_free_model import GlossFreeSLTModel


CHECKPOINT_PATH = "checkpoints/best.pt"
MAX_FRAMES = 256


def get_next_audio_path() -> str:
    """
    Tao ten file audio tu dong:
        output_speech_001.mp3
        output_speech_002.mp3
        output_speech_003.mp3
        ...

    Khong ghi de file audio da ton tai.
    """
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    index = 1

    while True:
        output_path = output_dir / f"output_speech_{index:03d}.mp3"

        if not output_path.exists():
            return str(output_path)

        index += 1


def pad_or_truncate(sequence: np.ndarray, max_len: int) -> np.ndarray:
    T = sequence.shape[0]

    if T >= max_len:
        return sequence[:max_len]

    pad = np.zeros((max_len - T, sequence.shape[1]))

    return np.concatenate([sequence, pad], axis=0)


def extract_from_video(video_path: str, pose_landmarker, hand_landmarker):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Khong mo duoc video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames_landmarks = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame = resize_frame(frame)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb,
        )

        timestamp_ms = int((frame_idx / fps) * 1000)

        pose_result = pose_landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        hand_result = hand_landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        frames_landmarks.append(
            extract_frame_vec(
                pose_result,
                hand_result,
            )
        )

        frame_idx += 1

    cap.release()

    sequence = np.array(frames_landmarks)

    if sequence.shape[0] == 0:
        raise ValueError("Khong doc duoc frame nao tu video.")

    sequence = normalize(sequence)

    sequence = pad_or_truncate(
        sequence,
        MAX_FRAMES,
    )

    return sequence.astype(np.float32)


async def speak(text: str, out_path: str):
    communicate = edge_tts.Communicate(
        text,
        voice="en-US-AriaNeural",
    )

    await communicate.save(out_path)


def main(video_path: str):

    os.makedirs("outputs", exist_ok=True)

    # --------------------------------------------------
    # Tao ten file audio moi, khong ghi de file cu
    # --------------------------------------------------
    output_audio = get_next_audio_path()

    latency = {}

    # --------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------
    print("--- Dang load model ---")

    t0 = time.perf_counter()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = GlossFreeSLTModel(
        unfreeze_last_n_blocks=2
    ).to(device)

    model.load_state_dict(
        torch.load(
            CHECKPOINT_PATH,
            map_location=device,
        )
    )

    model.eval()

    latency["load_model"] = time.perf_counter() - t0

    print(f"  ({latency['load_model']:.2f}s)")

    # --------------------------------------------------
    # MEDIAPIPE
    # --------------------------------------------------
    print(
        "--- Dang trich xuat pose tu video "
        "(MediaPipe, da toi uu toc do) ---"
    )

    t0 = time.perf_counter()

    pose_landmarker, hand_landmarker = build_landmarkers()

    pose_seq = extract_from_video(
        video_path,
        pose_landmarker,
        hand_landmarker,
    )

    pose_landmarker.close()
    hand_landmarker.close()

    latency["mediapipe_extraction"] = (
        time.perf_counter() - t0
    )

    print(
        f"  ({latency['mediapipe_extraction']:.2f}s)"
    )

    # --------------------------------------------------
    # MODEL INFERENCE
    # --------------------------------------------------
    print("--- Dang sinh cau tieng Anh ---")

    t0 = time.perf_counter()

    pose_tensor = (
        torch.tensor(pose_seq)
        .unsqueeze(0)
        .to(device)
    )

    generated_text = model.generate(
        pose_tensor
    )[0]

    latency["model_inference"] = (
        time.perf_counter() - t0
    )

    print(
        f"  ({latency['model_inference']:.2f}s)"
    )

    # --------------------------------------------------
    # TEXT TO SPEECH
    # --------------------------------------------------
    print("--- Dang chuyen thanh giong noi ---")

    t0 = time.perf_counter()

    asyncio.run(
        speak(
            generated_text,
            output_audio,
        )
    )

    latency["tts"] = time.perf_counter() - t0

    print(
        f"  ({latency['tts']:.2f}s)"
    )

    # --------------------------------------------------
    # RESULT
    # --------------------------------------------------
    print("\n=== KET QUA ===")

    print(
        f"Video: {video_path}"
    )

    print(
        f"Cau du doan: {generated_text}"
    )

    print(
        f"File am thanh: {output_audio}"
    )

    # --------------------------------------------------
    # LATENCY
    # --------------------------------------------------
    print("\n=== LATENCY (giay) ===")

    total = 0

    for stage, t in latency.items():

        print(
            f"  {stage}: {t:.3f}s"
        )

        if stage != "load_model":
            total += t

    print(
        f"  TONG (khong tinh load model): "
        f"{total:.3f}s"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    main(args.video)