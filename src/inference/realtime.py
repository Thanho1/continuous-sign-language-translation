"""
UC1: Dich realtime toi uu - MediaPipe chay NGAM trong luc dang quay (background thread + queue),
giam manh thoi gian cho sau khi bam dung. Giao dien nut bam kieu camera.

Chay: python -m src.inference.realtime --camera 0
"""
import argparse
import asyncio
import math
import os
import queue
import threading
import time

import cv2
import edge_tts
import mediapipe as mp
import numpy as np
import torch

from src.data.extract_keypoints import build_landmarkers, extract_frame_vec, normalize
from src.models.gloss_free_model import GlossFreeSLTModel

CHECKPOINT_PATH = "checkpoints/best.pt"
MAX_FRAMES = 256
OUTPUT_AUDIO = "outputs/realtime_speech.mp3"
WINDOW_NAME = "VSL Realtime Translation"
FONT = cv2.FONT_HERSHEY_SIMPLEX
DEBOUNCE_SECONDS = 0.8
EXTRACT_QUEUE_MAXSIZE = 3  # neu extraction cham hon camera, tu dong bo bot frame cu
DRAIN_TIMEOUT = 1.5  # thoi gian toi da cho hang doi xu ly not sau khi bam dung

state_lock = threading.Lock()
state = {
    "status": "idle",  # idle | recording | processing | done
    "record_start_time": None,
    "processing_start_time": None,
    "pose_vectors": [],       # ket qua MediaPipe da xu ly xong, tich luy dan trong luc quay
    "last_text": "",
    "last_toggle_time": 0.0,
}
running = True
frame_queue = queue.Queue(maxsize=EXTRACT_QUEUE_MAXSIZE)


def pad_or_truncate(sequence: np.ndarray, max_len: int) -> np.ndarray:
    T = sequence.shape[0]
    if T >= max_len:
        return sequence[:max_len]
    pad = np.zeros((max_len - T, sequence.shape[1]))
    return np.concatenate([sequence, pad], axis=0)


def extraction_worker(pose_landmarker, hand_landmarker, session_start_time):
    """Chay lien tuc tren background thread: lay frame tu queue, chay MediaPipe,
    tich luy ket qua vao state['pose_vectors'] MIEN LA dang o trang thai 'recording'."""
    while running:
        try:
            frame = frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        timestamp_ms = int((time.time() - session_start_time) * 1000)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        vec = extract_frame_vec(pose_result, hand_result)

        with state_lock:
            if state["status"] == "recording":
                state["pose_vectors"].append(vec)


async def speak_async(text: str, out_path: str):
    communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural")
    await communicate.save(out_path)


def finalize_and_translate(model, device):
    """Chay sau khi da drain queue - luc nay MediaPipe da lam xong phan lon,
    chi can chuan hoa + goi model + TTS."""
    with state_lock:
        vectors = list(state["pose_vectors"])

    if len(vectors) < 5:
        with state_lock:
            state["status"] = "idle"
        print("Qua it du lieu, bo qua.")
        return

    latency = {}
    t0 = time.perf_counter()
    sequence = np.array(vectors)
    sequence = normalize(sequence)
    sequence = pad_or_truncate(sequence, MAX_FRAMES).astype(np.float32)
    latency["chuan_hoa"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    pose_tensor = torch.tensor(sequence).unsqueeze(0).to(device)
    generated_text = model.generate(pose_tensor)[0]
    latency["model"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    os.makedirs("outputs", exist_ok=True)
    asyncio.run(speak_async(generated_text, OUTPUT_AUDIO))
    latency["tts"] = time.perf_counter() - t0

    print(f"Cau du doan: {generated_text}")
    print(f"So frame da trich xuat (chay ngam luc quay): {len(vectors)}")
    print(f"Latency SAU KHI DUNG (khong tinh MediaPipe, da chay ngam truoc do): "
          f"chuan_hoa={latency['chuan_hoa']:.2f}s | model={latency['model']:.2f}s | "
          f"tts={latency['tts']:.2f}s | TONG={sum(latency.values()):.2f}s")

    try:
        os.startfile(os.path.abspath(OUTPUT_AUDIO))
    except Exception as e:
        print(f"(Khong tu phat duoc am thanh: {e})")

    with state_lock:
        state["last_text"] = generated_text
        state["status"] = "idle"


def toggle_recording():
    now = time.time()
    with state_lock:
        if now - state["last_toggle_time"] < DEBOUNCE_SECONDS:
            return
        state["last_toggle_time"] = now

        if state["status"] in ("idle", "done"):
            state["status"] = "recording"
            state["pose_vectors"] = []
            state["record_start_time"] = now
            return

        if state["status"] == "recording":
            state["status"] = "draining"  # cho hang doi xu ly not vai frame con lai
            state["processing_start_time"] = now

    # Cho toi da DRAIN_TIMEOUT giay de queue xu ly not, roi chuyen sang buoc cuoi
    def drain_then_finish():
        deadline = time.time() + DRAIN_TIMEOUT
        while time.time() < deadline and not frame_queue.empty():
            time.sleep(0.05)
        with state_lock:
            state["status"] = "processing"
        finalize_and_translate(toggle_recording.model, toggle_recording.device)

    threading.Thread(target=drain_then_finish, daemon=True).start()


def request_quit():
    global running
    running = False


def wrap_text(text: str, max_chars: int = 55):
    words = text.split()
    lines, current = [], ""
    for w in words:
        if len(current) + len(w) + 1 <= max_chars:
            current = (current + " " + w).strip()
        else:
            lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines[:3]


def draw_ui(frame, buttons):
    h, w = frame.shape[:2]
    with state_lock:
        status = state["status"]
        record_start = state["record_start_time"]
        processing_start = state["processing_start_time"]
        last_text = state["last_text"]
        n_frames = len(state["pose_vectors"])

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 150), (w, h), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    if status in ("idle", "done"):
        text, color = "San sang", (0, 220, 0)
    elif status == "recording":
        elapsed = time.time() - record_start
        text, color = f"DANG QUAY  {elapsed:0.1f}s  (da xu ly {n_frames} frame)", (0, 0, 255)
    elif status == "draining":
        text, color = "Dang hoan tat trich xuat...", (0, 165, 255)
    else:
        elapsed = time.time() - processing_start
        text, color = f"DANG DICH...  {elapsed:0.1f}s", (0, 165, 255)
    cv2.putText(frame, text, (24, 44), FONT, 0.85, color, 2)

    if last_text:
        lines = wrap_text(last_text)
        box_h = 34 * len(lines) + 20
        sub_overlay = frame.copy()
        cv2.rectangle(sub_overlay, (0, h - 150 - box_h), (w, h - 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(sub_overlay, 0.5, frame, 0.5, 0)
        for i, line in enumerate(lines):
            y = h - 150 - box_h + 34 * (i + 1)
            cv2.putText(frame, line, (24, y), FONT, 0.75, (255, 255, 255), 2)

    cx, cy = buttons["record"]["center"]
    r = buttons["record"]["radius"]

    if status == "recording":
        pulse = int(6 * abs(math.sin(time.time() * 4)))
        cv2.circle(frame, (cx, cy), r + pulse, (0, 0, 255), 3)
        sq = 26
        cv2.rectangle(frame, (cx - sq // 2, cy - sq // 2), (cx + sq // 2, cy + sq // 2), (255, 255, 255), -1)
    elif status in ("processing", "draining"):
        cv2.circle(frame, (cx, cy), r, (0, 165, 255), 3)
        angle = int((time.time() * 220) % 360)
        cv2.ellipse(frame, (cx, cy), (r - 10, r - 10), 0, angle, angle + 110, (0, 165, 255), 5)
    else:
        cv2.circle(frame, (cx, cy), r, (255, 255, 255), 3)
        cv2.circle(frame, (cx, cy), r - 10, (0, 0, 255), -1)

    qx, qy = buttons["quit"]["center"]
    qr = buttons["quit"]["radius"]
    cv2.circle(frame, (qx, qy), qr, (255, 255, 255), 2)
    d = 9
    cv2.line(frame, (qx - d, qy - d), (qx + d, qy + d), (255, 255, 255), 2)
    cv2.line(frame, (qx - d, qy + d), (qx + d, qy - d), (255, 255, 255), 2)

    fx, fy = buttons["fullscreen"]["center"]
    fr = buttons["fullscreen"]["radius"]
    cv2.circle(frame, (fx, fy), fr, (255, 255, 255), 2)
    cv2.putText(frame, "F", (fx - 8, fy + 8), FONT, 0.6, (255, 255, 255), 2)

    hint = "Bam de quay" if status in ("idle", "done") else ("Bam de dung" if status == "recording" else "Vui long doi...")
    (tw, _), _ = cv2.getTextSize(hint, FONT, 0.55, 1)
    cv2.putText(frame, hint, (cx - tw // 2, h - 18), FONT, 0.55, (200, 200, 200), 1)

    return frame


def make_buttons(w, h):
    return {
        "record": {"center": (w // 2, h - 75), "radius": 44},
        "quit": {"center": (w - 50, 50), "radius": 24},
        "fullscreen": {"center": (60, 50), "radius": 24},
    }


def mouse_callback(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    buttons = param["buttons"]
    win_state = param["win_state"]

    def in_circle(px, py, center, radius):
        return (px - center[0]) ** 2 + (py - center[1]) ** 2 <= radius ** 2

    if in_circle(x, y, buttons["record"]["center"], buttons["record"]["radius"]):
        toggle_recording()
    elif in_circle(x, y, buttons["quit"]["center"], buttons["quit"]["radius"]):
        request_quit()
    elif in_circle(x, y, buttons["fullscreen"]["center"], buttons["fullscreen"]["radius"]):
        win_state["is_fullscreen"] = not win_state["is_fullscreen"]
        prop = cv2.WINDOW_FULLSCREEN if win_state["is_fullscreen"] else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, prop)


def main(camera_index: int, start_fullscreen: bool):
    global running
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("--- Dang load model ---")
    model = GlossFreeSLTModel(unfreeze_last_n_blocks=2).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    toggle_recording.model = model
    toggle_recording.device = device

    print("--- Dang khoi tao MediaPipe (chay 1 lan duy nhat cho ca phien) ---")
    pose_landmarker, hand_landmarker = build_landmarkers()
    session_start_time = time.time()  # moc thoi gian goc, dam bao timestamp luon tang dan

    worker_thread = threading.Thread(
        target=extraction_worker,
        args=(pose_landmarker, hand_landmarker, session_start_time),
        daemon=True,
    )
    worker_thread.start()

    print(f"--- Dang mo camera so {camera_index} ---")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc camera so {camera_index}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    win_state = {"is_fullscreen": start_fullscreen}
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if win_state["is_fullscreen"]:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("\n=== HUONG DAN ===")
    print("MediaPipe se chay NGAM ngay trong luc quay - bam dung se cho nhanh hon nhieu")
    print("Bam nut tron (hoac phim 's') de BAT DAU / DUNG quay")
    print("Bam nut F (hoac phim 'f') de fullscreen | Bam X (hoac 'q') de thoat\n")

    while running:
        ret, frame = cap.read()
        if not ret:
            print("Khong doc duoc frame tu camera.")
            break

        h, w = frame.shape[:2]
        buttons = make_buttons(w, h)
        cv2.setMouseCallback(WINDOW_NAME, mouse_callback, {"buttons": buttons, "win_state": win_state})

        with state_lock:
            is_recording = state["status"] == "recording"

        if is_recording:
            try:
                frame_queue.put_nowait(frame.copy())
            except queue.Full:
                pass  # bo qua frame nay neu extraction chua kip xu ly, giu camera muot

        display_frame = draw_ui(frame, buttons)
        cv2.imshow(WINDOW_NAME, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            toggle_recording()
        elif key == ord('f'):
            win_state["is_fullscreen"] = not win_state["is_fullscreen"]
            prop = cv2.WINDOW_FULLSCREEN if win_state["is_fullscreen"] else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, prop)
        elif key == ord('q'):
            running = False

    cap.release()
    cv2.destroyAllWindows()
    pose_landmarker.close()
    hand_landmarker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args()
    main(args.camera, args.fullscreen)