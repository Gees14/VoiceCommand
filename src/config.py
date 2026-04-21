"""
config.py

Central configuration for the voice commands classification project.
All paths, audio parameters, and model hyper-parameters live here so
that every other module can import a single source of truth.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory structure
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR       = PROJECT_ROOT / "data" / "raw" / "speech_commands"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR      = PROJECT_ROOT / "artifacts"
REPORTS_DIR        = PROJECT_ROOT / "reports"

# ---------------------------------------------------------------------------
# Audio parameters
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 16000          # Hz — matches Speech Commands dataset
DURATION: float  = 1.0            # seconds
N_SAMPLES: int   = int(SAMPLE_RATE * DURATION)   # 16 000 samples per clip

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

N_MFCC: int = 13   # number of MFCC coefficients

# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------

RANDOM_STATE: int  = 42
TEST_SPLIT: float  = 0.15   # fraction of total data held out for testing
VAL_SPLIT: float   = 0.15   # fraction of total data held out for validation

# ---------------------------------------------------------------------------
# Target classes (voice commands to recognise)
# ---------------------------------------------------------------------------

TARGET_CLASSES: list = ["left", "right", "up", "down", "stop"]

# ---------------------------------------------------------------------------
# Optional "unknown" catch-all class
# ---------------------------------------------------------------------------

INCLUDE_UNKNOWN: bool    = True
UNKNOWN_LABEL: str       = "unknown"
MAX_UNKNOWN_SAMPLES: int = 2000   # cap to avoid severe class imbalance

# ---------------------------------------------------------------------------
# Artifact file paths — command classifier
# ---------------------------------------------------------------------------

FEATURES_CSV          = PROCESSED_DATA_DIR / "features.csv"
DATASET_METADATA_JSON = PROCESSED_DATA_DIR / "dataset_metadata.json"

MODEL_PATH            = ARTIFACTS_DIR / "model.joblib"
LABEL_ENCODER_PATH    = ARTIFACTS_DIR / "label_encoder.joblib"
TRAIN_METADATA_PATH   = ARTIFACTS_DIR / "train_metadata.json"

CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"

# ---------------------------------------------------------------------------
# Speaker verification — paths
# ---------------------------------------------------------------------------

SPEAKERS_DIR        = PROJECT_ROOT / "data" / "speakers"
VERIFIER_PATH       = ARTIFACTS_DIR / "speaker_verifier.joblib"

# ---------------------------------------------------------------------------
# Speaker verification — GMM-UBM parameters
# ---------------------------------------------------------------------------

N_GMM_COMPONENTS: int        = 16     # mixture components per speaker model
GMM_COVARIANCE_TYPE: str     = "diag" # diagonal is standard for MFCC GMMs
MAP_RELEVANCE_FACTOR: float  = 16.0   # Reynolds et al. (2000) recommended value

# LLR threshold: raise to be stricter (fewer false accepts), lower to be looser.
# Run enroll.py --calibrate after enrollment to find the optimal value.
SPEAKER_THRESHOLD: float     = 0.0

# ---------------------------------------------------------------------------
# Enrollment recording parameters
# ---------------------------------------------------------------------------

ENROLLMENT_DURATION: float = 1.5          # seconds per recorded clip
N_ENROLLMENT_REPS: int     = 5            # repetitions per command
ENROLLMENT_COMMANDS: list  = TARGET_CLASSES  # same 5 commands → 25 clips/speaker
