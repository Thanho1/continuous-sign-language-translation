from pathlib import Path

from src.inference.video_inference import main


def run_demo():
    demo_videos = [
        "demo_092.mp4",
        "demo_181.mp4",
        "demo_234.mp4",
    ]

    for video in demo_videos:
        video_path = Path(video)

        if not video_path.exists():
            print(f"[WARNING] Khong tim thay: {video}")
            continue

        print("\n" + "=" * 80)
        print(f"DEMO: {video}")
        print("=" * 80)

        main(str(video_path))


if __name__ == "__main__":
    run_demo()