# Continuous Sign Language Translation

## 1. Giới thiệu

Dự án xây dựng hệ thống **nhận diện ngôn ngữ ký hiệu từ chuỗi chuyển động của cơ thể và bàn tay**, sau đó sinh ra câu văn bản tương ứng với nội dung của chuỗi ký hiệu.

Thay vì sử dụng trực tiếp ảnh/video RGB, dự án sử dụng **landmark/keypoint** được trích xuất từ cơ thể và hai bàn tay làm đầu vào cho mô hình.

### Pipeline tổng quát

```text
Video / Sign Language Sequence
            │
            ▼
     Landmark Extraction
            │
            ▼
Pose + Left Hand + Right Hand
            │
            ▼
     Remove Face Landmarks
            │
            ▼
   Normalize Landmark Coordinates
            │
            ▼
   Padding / Truncation to 256 frames
            │
            ▼
       225 features/frame
            │
            ▼
        Pose Encoder
            │
            ▼
        GPT-2 Decoder
            │
            ▼
     Generated Text Sentence
```

## 2. Dữ liệu

Dataset được biểu diễn dưới dạng các chuỗi landmark và câu văn bản tương ứng.

Mỗi sample gồm:
- `split`: tập dữ liệu (train, validation, test)
- `npy_path`: đường dẫn tới file landmark `.npy`
- `text`: câu văn bản Ground Truth tương ứng

Trong notebook EDA hiện tại, manifest cục bộ gồm:

| Split | Số sample |
|---|---:|
| Validation | 500 |
| Test | 500 |
| **Tổng** | **1,000** |

> **Lưu ý:** 1,000 sample trên là manifest validation/test được sử dụng để phân tích và đánh giá cục bộ. Các thí nghiệm huấn luyện với 14,000 training samples được thực hiện trước đó trong môi trường Colab/Kaggle và không nằm trong manifest cục bộ này.

## 3. Khám phá dữ liệu (EDA)

EDA được thực hiện trong notebook:

```
notebooks/EDA.ipynb
```

Các nội dung chính:

### 3.1. Kiểm tra cấu trúc dataset

Kiểm tra:
- Số lượng sample
- Các trường dữ liệu
- Kiểu dữ liệu
- Missing values
- Đường dẫn tới file landmark
- Ground Truth sentence

Manifest có 3 cột:
- `split`
- `npy_path`
- `text`

### 3.2. Kích thước dataset

Validation và Test hiện có:

```
Validation: 500 samples
Test:       500 samples
Total:      1000 samples
```

Training dataset được mở rộng qua nhiều thí nghiệm, từ 300 lên 2,000, 6,000, 12,000 và cuối cùng 14,000 samples.

### 3.3. Phân tích độ dài câu

Độ dài câu được tính bằng số lượng từ trong Ground Truth.

Kết quả trên validation/test:

| Split | Mean | Median | Std | Min | Max |
|---|---:|---:|---:|---:|---:|
| Test | 17.55 | 14.00 | 13.51 | 3 | 103 |
| Validation | 18.39 | 16.00 | 11.39 | 3 | 75 |

Quan sát chính:
- Độ dài câu trung bình khoảng 18 từ.
- Dataset chứa cả các câu ngắn và các chuỗi câu dài.
- Một số sample có độ dài lớn hơn đáng kể so với phần lớn dữ liệu.
- Vì vậy, bài toán không chỉ yêu cầu nhận diện nội dung ký hiệu mà còn yêu cầu mô hình học quan hệ giữa chuỗi chuyển động và câu văn bản có độ dài khác nhau.

## 4. Tiền xử lý và biểu diễn đầu vào

### 4.1. Landmark representation

Biểu diễn landmark ban đầu gồm:

```
Pose + Face + Left Hand + Right Hand
```

Trong đó:
- Pose: 33 landmarks
- Face: 468 landmarks
- Left hand: 21 landmarks
- Right hand: 21 landmarks

Tổng cộng:

```
33 + 468 + 21 + 21 = 543 landmarks
```

Mỗi landmark chứa 3 tọa độ: `(x, y, z)`

### 4.2. Loại bỏ Face landmarks

Để giảm số lượng đặc trưng và tập trung vào thông tin chuyển động của cơ thể và bàn tay, face landmarks được loại bỏ.

Giữ lại:
- 33 pose landmarks
- 21 left-hand landmarks
- 21 right-hand landmarks

Tổng:

```
33 + 21 + 21 = 75 landmarks
```

Mỗi landmark có 3 tọa độ `(x, y, z)`. Do đó số feature trên mỗi frame là:

```
75 × 3 = 225 features/frame
```

### 4.3. Chuẩn hóa

Các landmark được chuẩn hóa dựa trên shoulder center nhằm giảm ảnh hưởng của vị trí tương đối của người trong khung hình.

Mục tiêu của bước này là giúp mô hình tập trung hơn vào chuyển động và cấu trúc tương đối của cơ thể/bàn tay thay vì vị trí tuyệt đối trong không gian.

### 4.4. Padding và Truncation

Các sequence có thể có số lượng frame khác nhau.

Để đưa dữ liệu về cùng một kích thước đầu vào, sequence được:
- **Padding** nếu số frame nhỏ hơn giới hạn.
- **Truncation** nếu số frame vượt quá giới hạn.

Chiều dài tối đa: `256 frames`

Do đó biểu diễn cuối cùng của mỗi sample có dạng:

```
(256, 225)
```

Tức là: 256 frames × 225 features/frame

### 4.5. Pipeline preprocessing

```text
Raw Landmark Sequence
        │
        ▼
Remove Face Landmarks
        │
        ▼
Keep Pose + Left Hand + Right Hand
        │
        ▼
Normalize using Shoulder Center
        │
        ▼
Pad / Truncate to 256 Frames
        │
        ▼
Save as .npy
        │
        ▼
Pose Encoder
        │
        ▼
GPT-2 Decoder
        │
        ▼
Generated Sentence
```

## 5. Kiến trúc mô hình

Hệ thống gồm hai thành phần chính:

```
Pose Encoder → GPT-2 Decoder
```

### 5.1. Pose Encoder

Input của encoder là chuỗi landmark:

```
(batch_size, 256, 225)
```

Trong đó:
- 256: số frame tối đa của sequence
- 225: số feature trên mỗi frame

Encoder có nhiệm vụ chuyển chuỗi chuyển động thành biểu diễn đặc trưng để cung cấp cho decoder.

### 5.2. GPT-2 Decoder

GPT-2 được sử dụng để sinh câu văn bản từ biểu diễn được tạo bởi pose encoder.

Quá trình tổng quát:

```text
Landmark Sequence
       ↓
Pose Encoder
       ↓
Motion Representation
       ↓
GPT-2 Decoder
       ↓
Generated Sentence
```

## 6. Các thí nghiệm huấn luyện

Các thí nghiệm được thực hiện trước đó trong môi trường Colab/Kaggle.

Mục tiêu chính của các thí nghiệm là quan sát ảnh hưởng của việc tăng kích thước training dataset đến khả năng sinh văn bản của mô hình.

### 6.1. Bảng kết quả

| Lần | Train | Validation | Test | Best Val Loss | BLEU ↑ | WER ↓ | Ghi chú |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 300 | 60 | 60 | 3.2100 | 0.48 | 1.0445 | 15 epochs |
| 2 | 2,000 | 200 | 200 | 1.8784 | – | – | Best ở epoch 6 |
| 3 | 6,000 | 400 | 400 | 1.8571 | 0.87 | 1.2150 | Dataset tăng đáng kể |
| 4 | 12,000 | 400 | 400 | 1.8240 | 1.67 | 1.0894 | BLEU tăng mạnh |
| 5 | 14,000 | 500 | 500 | 1.9351 | 1.74 | 1.0907 | MAX_FRAMES: 150 → 256 |

Các kết quả trên là kết quả được ghi nhận từ các lần training trước, không được re-run trong notebook EDA hiện tại.

### 6.2. Nhận xét

Khi tăng kích thước training dataset:
- Validation loss giảm mạnh từ thí nghiệm 300 samples xuống các thí nghiệm có dataset lớn hơn.
- BLEU tăng từ 0.48 lên 1.74.

Kết quả tốt nhất về BLEU trong các thí nghiệm được ghi nhận là:

```
BLEU = 1.74
```

với:
- Train = 14,000 samples
- Validation = 500 samples
- Test = 500 samples

WER tương ứng:

```
WER = 1.0907
```

Thí nghiệm cuối cùng cũng thay đổi `MAX_FRAMES: 150 → 256` để cho phép mô hình xử lý các sequence dài hơn.

## 7. Đánh giá mô hình

### 7.1. BLEU

BLEU được sử dụng để đánh giá mức độ tương đồng giữa câu mô hình sinh ra và câu Ground Truth dựa trên sự trùng khớp của các n-gram.

Trong thí nghiệm cuối:

```
BLEU = 1.74
```

### 7.2. WER

WER (Word Error Rate) đo số lượng lỗi cần thực hiện để biến câu Prediction thành Ground Truth, bao gồm:
- Substitution
- Deletion
- Insertion

Công thức tổng quát:

```
WER = (S + D + I) / N
```

Trong đó:
- S: số từ bị thay thế
- D: số từ bị xóa
- I: số từ bị chèn
- N: số từ trong Ground Truth

Kết quả thí nghiệm cuối:

```
WER = 1.0907
```

## 8. Semantic Similarity (SIM)

Ngoài BLEU và WER, dự án sử dụng Semantic Similarity để đánh giá mức độ tương đồng về ngữ nghĩa giữa Ground Truth và Prediction.

SIM được tính bằng cách:
1. Encode Ground Truth thành sentence embedding.
2. Encode Prediction thành sentence embedding.
3. Tính cosine similarity giữa hai embedding.

Giá trị cosine similarity càng lớn thì hai câu càng có xu hướng gần nhau về mặt ngữ nghĩa.

SIM được sử dụng như một metric bổ sung cho BLEU và WER, không thay thế hai metric này.

### 8.1. Kết quả SIM trên 500 test samples

```
Number of samples: 500
Missing SIM: 0

Mean SIM   = 0.1305
Median SIM = 0.1235
Min SIM    = -0.0758
Max SIM    = 0.5376
```

Trong đó Mean SIM được sử dụng làm chỉ số chính khi nhận xét tổng quan về khả năng giữ ngữ nghĩa của mô hình.

### 8.2. Top 5 và Bottom 5 SIM

Notebook cũng lưu lại các sample có:
- SIM cao nhất
- SIM thấp nhất

nhằm kiểm tra định tính sự khác biệt giữa Ground Truth và Prediction.

Ví dụ một sample có SIM cao:

```
GT:
Apply that foundation, all over the face.

Prediction:
So, you're going to want to make sure that you have a good foundation.

SIM:
0.5376
```

Một số sample có SIM thấp cho thấy Prediction có thể không giữ được nội dung/ngữ nghĩa của Ground Truth:

```
GT:
You can see that they have got the wool liners inside of them...

Prediction:
So, you're going to want to make sure that you have the right tools...

SIM:
-0.0758
```

Các trường hợp này được sử dụng để minh họa giới hạn của mô hình trong việc sinh đúng nội dung câu.

## 9. Kết quả đánh giá cuối cùng

Trên tập test gồm 500 samples:

```
BLEU = 1.74
WER  = 1.0907
```

Kết quả Semantic Similarity:

```
Mean SIM   = 0.1305
Median SIM = 0.1235
Min SIM    = -0.0758
Max SIM    = 0.5376
```

Một số prediction được lưu tại:

```
outputs/predictions.csv
```

File chứa:
- Ground Truth
- Prediction
- SIM

cho 500 test samples.

## 10. Cấu trúc project

Cấu trúc thư mục tổng quát:

```
project/
│
├── data/
│   ├── processed/
│   │   └── keypoints_*/
│   │       └── *.npy
│   │
│   └── ...
│
├── notebooks/
│   └── EDA.ipynb
│
├── outputs/
│   └── predictions.csv
│
├── ...
│
└── README.md
```

Cấu trúc có thể thay đổi tùy theo môi trường training và cách tổ chức dataset thực tế.

## 11. Cài đặt môi trường

Khuyến nghị sử dụng: Python 3.x

Các thư viện chính được sử dụng trong quá trình phân tích và đánh giá gồm:

```bash
pip install numpy pandas matplotlib scikit-learn
pip install sentence-transformers
```

Nếu chạy notebook:

```bash
pip install jupyter
```

hoặc có thể chạy trực tiếp bằng:
- Jupyter Notebook
- JupyterLab
- VS Code Notebook

## 12. Chạy EDA Notebook

Mở terminal tại thư mục project:

```bash
jupyter notebook
```

Sau đó mở:

```
notebooks/EDA.ipynb
```

và chạy các cell theo thứ tự từ trên xuống dưới.

Notebook thực hiện:

```text
Load manifest
    ↓
Dataset overview
    ↓
Dataset size
    ↓
Sentence length analysis
    ↓
Keypoint shape checking
    ↓
Input representation
    ↓
Preprocessing description
    ↓
Training experiment summary
    ↓
BLEU / WER results
    ↓
Semantic Similarity
    ↓
Top 5 / Bottom 5 SIM
```

## 13. Lưu ý khi chạy notebook

Các training experiments với 300, 2,000, 6,000, 12,000, 14,000 samples đã được thực hiện trước đó trong môi trường Colab/Kaggle.

Notebook hiện tại chủ yếu dùng để:
- phân tích dữ liệu;
- kiểm tra preprocessing;
- kiểm tra input representation;
- tổng hợp kết quả training;
- phân tích prediction;
- tính Semantic Similarity.

Do đó không cần chạy lại toàn bộ quá trình training để xem kết quả được báo cáo trong notebook.

## 14. Kết luận

Dự án xây dựng pipeline từ chuỗi landmark của cơ thể và hai bàn tay đến câu văn bản:

```text
Landmarks
   ↓
Preprocessing
   ↓
Pose Encoder
   ↓
GPT-2 Decoder
   ↓
Text Generation
```

Việc tăng kích thước training dataset từ 300 lên 14,000 samples giúp cải thiện đáng kể khả năng sinh văn bản, thể hiện qua sự tăng của BLEU từ:

```
0.48 → 1.74
```

Trong thí nghiệm cuối cùng trên 500 test samples:

```
BLEU = 1.74
WER  = 1.0907
```

Semantic Similarity trung bình đạt:

```
Mean SIM = 0.1305
```

Các kết quả cho thấy mô hình đã có khả năng sinh ra văn bản từ chuỗi chuyển động, tuy nhiên vẫn còn nhiều trường hợp Prediction khác đáng kể so với Ground Truth. Đây là cơ sở để tiếp tục nghiên cứu về chất lượng biểu diễn chuyển động, kích thước dataset và khả năng sinh ngôn ngữ của mô hình.
