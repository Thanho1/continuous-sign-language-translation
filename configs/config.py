# ==========================
# Camera Settings
# ==========================
CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ==========================
# Dataset Settings
# ==========================
ACTIONS = [
    "thanks",
    "sorry"
]

NUM_SEQUENCES = 20
SEQUENCE_LENGTH = 60
START_DELAY = 3

# ==========================
# Feature Settings
# ==========================
USE_FACE = False

POSE_DIM = 132
FACE_DIM = 1404
HAND_DIM = 63

NUM_FEATURES = POSE_DIM + HAND_DIM + HAND_DIM

if USE_FACE:
    NUM_FEATURES += FACE_DIM

FEATURE_DIM = NUM_FEATURES

# ==========================
# Train Settings
# ==========================
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ==========================
# Model Settings
# ==========================
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
DROPOUT = 0.3

# ==========================
# Paths
# ==========================
DATA_PATH = "data/processed"
LABEL_MAP_PATH = "checkpoints/label_map.json"

# ==========================
# Training
# ==========================

MODEL_PATH = "checkpoints/best_model.keras"

EPOCHS = 50

BATCH_SIZE = 8

LSTM_HIDDEN_SIZE = 128

DROPOUT = 0.3