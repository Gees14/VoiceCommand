"""
predict.py

Two-stage voice command inference:

    Stage 1 — Speaker Verification (GMM-UBM)
        Checks whether the voice belongs to one of the 4 enrolled speakers.
        Rejects audio from unrecognised voices before any command is decoded.

    Stage 2 — Command Classification (XGBoost pipeline)
        Classifies the audio into one of the target commands.
        Only runs if Stage 1 succeeds.

Usage:
    python src/predict.py path/to/audio.wav

Options:
    --skip-verification    Bypass speaker check (useful for testing commands only).
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import LABEL_ENCODER_PATH, MODEL_PATH
from prepare_dataset import extract_features, load_audio
from speaker_verifier import SpeakerVerifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-stage voice command recognition (speaker + command).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/predict.py recordings/alice_left.wav\n"
            "  python src/predict.py audio.wav --skip-verification\n"
        ),
    )
    parser.add_argument(
        "wav_file",
        type=str,
        help="Path to the .wav file to classify.",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Bypass speaker verification (run command classification only).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------


def load_command_artifacts():
    """Load the command classifier pipeline and LabelEncoder."""
    for path, name in [
        (MODEL_PATH,         "command model"),
        (LABEL_ENCODER_PATH, "label encoder"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find {name} at:\n  {path}\n"
                "Run train.py first."
            )
    model = joblib.load(MODEL_PATH)
    le    = joblib.load(LABEL_ENCODER_PATH)
    return model, le


# ---------------------------------------------------------------------------
# Stage 1 — Speaker verification
# ---------------------------------------------------------------------------


def run_speaker_verification(audio: np.ndarray) -> tuple:
    """
    Verify the speaker identity using the GMM-UBM verifier.

    Returns:
        (is_authorized, speaker_name, llr_score)
    """
    verifier = SpeakerVerifier.load()
    return verifier.verify(audio)


# ---------------------------------------------------------------------------
# Stage 2 — Command classification
# ---------------------------------------------------------------------------


def run_command_classification(audio: np.ndarray, model, le) -> tuple:
    """
    Classify the voice command using the XGBoost pipeline.

    Returns:
        (best_label, confidence, top3_list)
        where top3_list = [(label, probability), ...]
    """
    features = extract_features(audio).reshape(1, -1)
    probas   = model.predict_proba(features)[0]
    top3_idx = np.argsort(probas)[::-1][:3]

    best_label  = le.classes_[top3_idx[0]]
    confidence  = float(probas[top3_idx[0]])
    top3        = [(le.classes_[i], float(probas[i])) for i in top3_idx]

    return best_label, confidence, top3


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _bar(probability: float, width: int = 22) -> str:
    """Render a simple ASCII probability bar."""
    filled = int(probability * width)
    return "█" * filled + "░" * (width - filled)


def print_result(
    wav_path: Path,
    speaker_name: str,
    llr_score: float,
    best_label: str,
    confidence: float,
    top3: list,
) -> None:
    """Print a formatted two-stage result card."""
    w = 52
    print()
    print("=" * w)
    print(f"  File            : {wav_path.name}")
    print(f"  Speaker         : {speaker_name.upper()}  (LLR {llr_score:+.3f})")
    print(f"  Command         : {best_label.upper()}")
    print(f"  Confidence      : {confidence * 100:.2f}%")
    print("-" * w)
    print("  Top-3 Predictions:")
    for rank, (label, prob) in enumerate(top3, start=1):
        print(f"    {rank}. {label:<12}  {prob*100:5.2f}%  {_bar(prob)}")
    print("=" * w)
    print()


def print_rejected(wav_path: Path, speaker_name: str, llr_score: float) -> None:
    """Print a rejection card when speaker verification fails."""
    w = 52
    print()
    print("=" * w)
    print(f"  File            : {wav_path.name}")
    print(f"  RESULT          : ACCESS DENIED")
    print(f"  Closest match   : {speaker_name}  (LLR {llr_score:+.3f})")
    print(f"  Reason          : Score below authorization threshold.")
    print("=" * w)
    print()


# ---------------------------------------------------------------------------
# Main inference pipeline
# ---------------------------------------------------------------------------


def predict_file(wav_path: Path, skip_verification: bool) -> None:
    """
    Run the full two-stage pipeline on a single WAV file.

    Args:
        wav_path:           Path to the input audio file.
        skip_verification:  If True, skip Stage 1 and use "unverified" as speaker.
    """
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")
    if wav_path.suffix.lower() != ".wav":
        raise ValueError(f"Expected a .wav file, got: {wav_path.suffix!r}")

    audio = load_audio(wav_path)
    if audio is None:
        raise RuntimeError(f"Failed to decode audio: {wav_path}")

    # ---- Stage 1: Speaker Verification ----
    if skip_verification:
        is_authorized = True
        speaker_name  = "unverified"
        llr_score     = 0.0
        logger.info("Speaker verification skipped (--skip-verification).")
    else:
        logger.info("Stage 1: Speaker verification …")
        is_authorized, speaker_name, llr_score = run_speaker_verification(audio)

        if not is_authorized:
            print_rejected(wav_path, speaker_name, llr_score)
            return   # stop here — do not classify commands for unknown speakers

        logger.info(
            "Speaker authorized: '%s'  (LLR %.3f)", speaker_name, llr_score
        )

    # ---- Stage 2: Command Classification ----
    logger.info("Stage 2: Command classification …")
    model, le = load_command_artifacts()
    best_label, confidence, top3 = run_command_classification(audio, model, le)

    print_result(wav_path, speaker_name, llr_score, best_label, confidence, top3)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args     = parse_args()
    wav_path = Path(args.wav_file).resolve()

    try:
        predict_file(wav_path, args.skip_verification)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
