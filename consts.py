from pathlib import Path

RANDOM_SEED = 42

DATA_DIR = Path("/scratch/nlp_G1/")
TEACHER_DIR = DATA_DIR / "models/teacher"
STUDENT_DIR = DATA_DIR / "models/student"
TOKENIZER_DIR = DATA_DIR / "models/tokenizer"

TOKENIZER_PATH = TOKENIZER_DIR / "gpt-clean-16000.json"


TRAIN_DATASET_STRICT_PATH = DATA_DIR / "data/train_10M_clean"
DEV_DATASET_STRICT_PATH = DATA_DIR / "data/dev_clean"