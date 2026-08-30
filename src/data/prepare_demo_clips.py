"""
Ghep test_video_ids.csv voi predictions.csv, chon 10 cau CO DO KHOP CAO
(GT va Pred co nhieu tu trung nhau) va DO DAI VUA PHAI (6-20 tu) -
day la bang chung thuyet phuc hon la chon cau ngan nhat (de bi mode collapse).

Chay: python -m src.data.prepare_demo_clips
"""
import csv
import os
import subprocess

from huggingface_hub import hf_hub_download

METADATA_REPO = "PSewmuthu/How2Sign_Holistic"
METADATA_FILE = "how2sign_holistic_features/metadata/how2sign_realigned_test.csv"

VIDEO_IDS_CSV = "data/processed/test_video_ids.csv"
PREDICTIONS_CSV = "outputs/predictions.csv"
OUT_DIR = "outputs/demo_clips"
N_CLIPS = 10

MIN_WORDS = 6
MAX_WORDS = 20
MAX_CLIP_DURATION = 10  # giay, tranh clip qua dai kho demo


def load_video_ids():
    with open(VIDEO_IDS_CSV, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_predictions():
    with open(PREDICTIONS_CSV, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_metadata():
    path = hf_hub_download(repo_id=METADATA_REPO, repo_type="dataset", filename=METADATA_FILE)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return {row["SENTENCE_NAME"]: row for row in reader}


def word_overlap_score(gt: str, pred: str) -> float:
    """Ty le tu trong GT cung xuat hien trong Pred (khong phan biet hoa/thuong)."""
    gt_words = set(w.strip(".,!?").lower() for w in gt.split())
    pred_words = set(w.strip(".,!?").lower() for w in pred.split())
    if not gt_words:
        return 0.0
    return len(gt_words & pred_words) / len(gt_words)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    video_ids = load_video_ids()
    predictions = load_predictions()
    metadata = load_metadata()

    assert len(video_ids) == len(predictions), (
        f"So dong lech nhau: video_ids={len(video_ids)}, predictions={len(predictions)}."
    )

    candidates = []
    seen_youtube_ids = set()

    for vid_row, pred_row in zip(video_ids, predictions):
        vid = vid_row["video_id"]
        if vid not in metadata:
            continue
        meta = metadata[vid]
        gt = pred_row["Ground Truth"]
        pred = pred_row["Prediction"]
        n_words = len(gt.split())

        if not (MIN_WORDS <= n_words <= MAX_WORDS):
            continue

        duration = float(meta["END_REALIGNED"]) - float(meta["START_REALIGNED"])
        if duration > MAX_CLIP_DURATION:
            continue

        youtube_id = meta["VIDEO_ID"]
        if youtube_id in seen_youtube_ids:
            continue  # tranh lay nhieu clip tu cung 1 video (giam trung lap do loi align)

        score = word_overlap_score(gt, pred)
        candidates.append({
            "youtube_id": youtube_id,
            "start": float(meta["START_REALIGNED"]),
            "end": float(meta["END_REALIGNED"]),
            "gt": gt,
            "pred": pred,
            "n_words": n_words,
            "overlap_score": round(score, 2),
        })
        seen_youtube_ids.add(youtube_id)

    # Uu tien do khop cao nhat truoc, cau it tu hon trong khoang cho phep xep sau
    candidates.sort(key=lambda c: (-c["overlap_score"], c["n_words"]))
    chosen = candidates[:N_CLIPS]

    print(f"Da chon {len(chosen)} clip (uu tien do khop tu cao, {MIN_WORDS}-{MAX_WORDS} tu):\n")
    for i, c in enumerate(chosen):
        print(f"[{i+1}] Do khop: {c['overlap_score']*100:.0f}%")
        print(f"    GT:   {c['gt']}")
        print(f"    PRED: {c['pred']}")
        print(f"    YouTube: https://youtube.com/watch?v={c['youtube_id']}  "
              f"({c['start']:.1f}s-{c['end']:.1f}s)\n")

    for i, c in enumerate(chosen):
        out_path = os.path.join(OUT_DIR, f"clip_{i+1:02d}.mp4")
        url = f"https://www.youtube.com/watch?v={c['youtube_id']}"
        section = f"*{c['start']:.2f}-{c['end']:.2f}"
        cmd = ["yt-dlp", "--download-sections", section, "-f", "mp4", "-o", out_path, url]
        print(f"Dang tai clip {i+1}/{len(chosen)}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  LOI: {result.stderr[-500:]}")
        else:
            print(f"  Da luu: {out_path}")

    with open(os.path.join(OUT_DIR, "chosen_clips_info.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["youtube_id", "start", "end", "gt", "pred", "n_words", "overlap_score"]
        )
        writer.writeheader()
        writer.writerows(chosen)

    print(f"\nHoan tat. Xem bang thong tin tai: {OUT_DIR}/chosen_clips_info.csv")


if __name__ == "__main__":
    main()