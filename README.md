# Continuous Sign Language Translation

A gloss-free approach for **Continuous Sign Language Translation** from video to natural-language text and speech.

## Demo

The project supports video-based inference and speech generation.

Demo videos and generated outputs are stored locally and are **not included in Git** because they are covered by `.gitignore`.

## Pipeline

```text
Input video
    │
    ▼
src/data/extract_keypoints.py
    │
    ▼
Pose / hand keypoints
    │
    ▼
src/models/pose_encoder.py
    │
    ▼
Gloss-free translation model
    │
    ▼
Natural-language text
    │
    ▼
Speech output
```

## Tech Stack

- **Python** - main programming language
- **OpenCV** - video processing and frame handling
- **MediaPipe** - pose and hand landmark extraction
- **NumPy** - numerical data processing
- **PyTorch** - model training and inference
- **FFmpeg** - video/audio processing
- **Pandas / Parquet** - metadata and tabular data processing
- **Text-to-Speech** - generation of speech from translated text

## Cấu trúc thư mục

```text
Project/
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── configs/
│   ├── config.py
│   └── paths.py
├── data/
│   ├── processed/
│   │   ├── manifest.csv
│   │   ├── test_video_ids.csv
│   │   └── keypoints_bdanko/
│   │       ├── train_*.npy
│   │       ├── validation_*.npy
│   │       └── test_*.npy
│   ├── raw/
│   │   └── transcripts.csv
│   └── splits/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── demo/
│   ├── demo_092.mp4
│   ├── demo_181.mp4
│   └── demo_234.mp4
├── models/
│   ├── hand_landmarker.task
│   ├── pose_landmarker_full.task
│   └── pose_landmarker_lite.task
├── notebooks/
├── outputs/
│   ├── predictions.csv
│   ├── output_speech_*.mp3
│   ├── realtime_speech.mp3
│   ├── extraction_visualized.mp4
│   ├── demo_clips/
│   ├── logs/
│   ├── predictions/
│   └── videos/
├── src/
│   ├── data/
│   │   ├── build_manifest_bdanko.py
│   │   ├── build_test_video_ids.py
│   │   ├── dataset.py
│   │   ├── extract_keypoints.py
│   │   └── prepare_demo_clips.py
│   ├── inference/
│   │   ├── realtime.py
│   │   ├── video_inference.py
│   │   └── visualize_extraction.py
│   ├── models/
│   │   ├── gloss_free_model.py
│   │   └── pose_encoder.py
│   ├── utils/
│   │   ├── latency.py
│   │   ├── metrics.py
│   │   ├── seed.py
│   │   └── visualize.py
│   ├── evaluate.py
│   ├── main.py
│   ├── train.py
│   └── tune_decoding.py
├── tests/
│   ├── find_best_demo.py
│   ├── find_best_samples.py
│   └── find_good_samples.py
├── ffmpeg.exe
├── metadata.parquet
├── README.md
├── requirements.txt
├── start_project.bat
├── test_one_sample.py
├── test_pipeline.py
└── test_video.mp4
```

### Lưu ý về dữ liệu và file lớn

Các thư mục/file sinh ra trong quá trình xử lý dữ liệu hoặc inference được đưa vào `.gitignore`, bao gồm:

- `data/raw/`
- `data/processed/`
- `data/splits/`
- `*.parquet`
- `checkpoints/`
- `outputs/`
- các file video (`*.mp4`, `*.avi`, `*.mov`, ...)
- các file audio (`*.mp3`, `*.wav`, ...)
- `ffmpeg.exe`

Thư mục `data/raw/videos/` đã được loại bỏ vì không chứa dữ liệu cần thiết.

## Installation

### 1. Clone project

```bash
git clone <repository-url>
cd Project
```

### 2. Tạo virtual environment

```bash
python -m venv vsl
```

Windows PowerShell:

```powershell
.\vsl\Scripts\Activate.ps1
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

## Data Preparation

Dữ liệu được tổ chức thành:

- `data/raw/` - raw metadata, hiện có `transcripts.csv`
- `data/splits/` - danh sách train/validation/test
- `data/processed/` - manifest và keypoints đã xử lý
- `data/processed/keypoints_bdanko/` - các file `.npy` theo từng split

Các script chính:

```text
src/data/build_manifest_bdanko.py
src/data/build_test_video_ids.py
src/data/extract_keypoints.py
src/data/dataset.py
src/data/prepare_demo_clips.py
```

## Model

Các thành phần model chính nằm trong `src/models/`:

- `pose_encoder.py` - xử lý/encode thông tin pose keypoints.
- `gloss_free_model.py` - mô hình Continuous Sign Language Translation theo hướng gloss-free.

Checkpoint:

```text
checkpoints/
├── best.pt
└── last.pt
```

## Training

File training chính:

```text
src/train.py
```

Lệnh chạy:

```bash
python -m src.train
```

Cấu hình đường dẫn/tham số:

```text
configs/config.py
configs/paths.py
```

## Evaluation

File đánh giá:

```text
src/evaluate.py
```

Chạy:

```bash
python -m src.evaluate
```

Các metric hỗ trợ nằm trong:

```text
src/utils/metrics.py
```

## Inference

### Video inference

```text
src/inference/video_inference.py
```

Dùng để chạy model trên video.

### Real-time inference

```text
src/inference/realtime.py
```

Dùng cho inference theo thời gian thực.

### Visualize keypoint extraction

```text
src/inference/visualize_extraction.py
```

Dùng để trực quan hóa quá trình trích xuất keypoints.

## Testing

Các script kiểm thử:

```text
test_one_sample.py
test_pipeline.py
```

Có thể chạy:

```bash
python test_one_sample.py
python test_pipeline.py
```

Các script hỗ trợ tìm sample/demo:

```text
tests/find_best_demo.py
tests/find_best_samples.py
tests/find_good_samples.py
```

## Outputs

Các kết quả sinh ra trong quá trình chạy project được lưu trong `outputs/`.

Ví dụ:

```text
outputs/
├── predictions.csv
├── output_speech_001.mp3
├── output_speech_002.mp3
├── ...
├── realtime_speech.mp3
├── extraction_visualized.mp4
├── demo_clips/
├── logs/
├── predictions/
└── videos/
```

Các file video/audio/output này được ignore và không commit lên Git.

## Utility Scripts

- `src/utils/latency.py` - hỗ trợ đo latency.
- `src/utils/metrics.py` - các hàm metric/evaluation.
- `src/utils/seed.py` - thiết lập random seed.
- `src/utils/visualize.py` - hỗ trợ trực quan hóa.
- `src/tune_decoding.py` - hỗ trợ tuning decoding.
- `src/main.py` - entry point tổng quát của project.

## Model Files

Project sử dụng các MediaPipe task model trong `models/`:

```text
models/
├── hand_landmarker.task
├── pose_landmarker_full.task
└── pose_landmarker_lite.task
```

Các file này được sử dụng cho quá trình landmark/keypoint extraction.

## Git & Repository

Các file không phù hợp để commit trực tiếp được loại khỏi Git thông qua `.gitignore`, đặc biệt là:

```text
checkpoints/
data/raw/
data/processed/
data/splits/
outputs/
*.parquet
*.mp4
*.avi
*.mov
*.mkv
*.webm
*.mp3
*.wav
*.flac
ffmpeg.exe
```

Kiểm tra trạng thái Git:

```bash
git status
```

Kiểm tra các file đang bị ignore:

```bash
git status --short --ignored
```

Kiểm tra một file cụ thể:

```bash
git check-ignore -v <file>
```

## Limitations & Future Work

Một số hướng phát triển:

- Cải thiện chất lượng Continuous Sign Language Translation.
- Tối ưu pose/hand keypoint representation.
- Cải thiện tốc độ inference và latency.
- Tối ưu decoding để tạo câu tự nhiên hơn.
- Mở rộng khả năng xử lý video trong điều kiện thực tế.
- Đánh giá mô hình trên nhiều điều kiện dữ liệu khác nhau.
- Hoàn thiện pipeline từ video → text → speech theo thời gian thực.

## License

MIT
