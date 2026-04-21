"""
enroll.py

Reads enrollment recordings from data/speakers/, trains the UBM on the
combined speaker pool, MAP-adapts a per-speaker GMM for each person, and
saves the complete SpeakerVerifier to artifacts/speaker_verifier.joblib.

Run from the project root after recording with record_samples.py:
    python src/enroll.py

Flags:
    --skip-ubm     Reuse an existing UBM (faster when adding a new speaker).
    --threshold N  Override the LLR decision threshold (float, default 0.0).
    --calibrate    Print LLR scores for every enrolled clip (helps set threshold).
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import SPEAKERS_DIR
from prepare_dataset import load_audio
from speaker_verifier import SpeakerVerifier, extract_speaker_features

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
        description="Build GMM-UBM speaker profiles from enrollment recordings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Typical usage:\n"
            "  # First time — train UBM and enroll all speakers:\n"
            "  python src/enroll.py\n\n"
            "  # Add one new speaker without retraining the UBM:\n"
            "  python src/enroll.py --skip-ubm\n\n"
            "  # Check what LLR scores look like to tune the threshold:\n"
            "  python src/enroll.py --calibrate\n"
        ),
    )
    parser.add_argument(
        "--skip-ubm",
        action="store_true",
        help="Load the existing UBM instead of retraining it.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Set LLR decision threshold (overrides config default of 0.0).",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="After enrollment print per-file LLR scores to help tune --threshold.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_speaker_features(speaker_dir: Path) -> List[np.ndarray]:
    """
    Load all WAV files under a speaker directory and extract frame-level MFCCs.

    Files that cannot be decoded are skipped with a warning.

    Returns:
        List of arrays, each with shape (n_frames, N_MFCC).
    """
    wav_files = sorted(speaker_dir.glob("*.wav"))
    if not wav_files:
        logger.warning("No WAV files in %s — skipping this speaker.", speaker_dir)
        return []

    features: List[np.ndarray] = []
    for wav_path in wav_files:
        audio = load_audio(wav_path)
        if audio is None:
            continue
        features.append(extract_speaker_features(audio))

    total_frames = sum(f.shape[0] for f in features)
    logger.info(
        "  %-16s  %d files  |  %d frames",
        speaker_dir.name, len(features), total_frames,
    )
    return features


def collect_all_speakers(root: Path) -> Dict[str, List[np.ndarray]]:
    """
    Walk the speakers root and return a mapping of name → MFCC matrices.

    Raises FileNotFoundError if the root directory does not exist.
    Raises RuntimeError if no valid speaker data is found.
    """
    if not root.exists():
        raise FileNotFoundError(
            f"Speakers directory not found: {root}\n"
            "Record enrollment samples first:\n"
            "  python src/record_samples.py --speaker <name>"
        )

    data: Dict[str, List[np.ndarray]] = {}
    for speaker_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        feats = load_speaker_features(speaker_dir)
        if feats:
            data[speaker_dir.name] = feats

    if not data:
        raise RuntimeError(
            f"No valid speaker data found in {root}.\n"
            "Run record_samples.py for each speaker first."
        )
    return data


# ---------------------------------------------------------------------------
# Calibration report
# ---------------------------------------------------------------------------


def print_calibration_report(verifier: SpeakerVerifier, speaker_data: Dict[str, List[np.ndarray]]) -> None:
    """
    Print LLR scores for every enrollment file.

    Each file is scored against ALL enrolled speaker models. This lets you
    see the score distribution and choose an appropriate threshold:
        - Correct-speaker scores should be positive (ideally >> 0)
        - Wrong-speaker scores should be negative (ideally << 0)
    """
    print()
    print("=" * 65)
    print("  CALIBRATION REPORT — LLR scores (true speaker first)")
    print("=" * 65)

    for true_name, feat_list in speaker_data.items():
        print(f"\n  Speaker: {true_name.upper()}")
        print(f"  {'File':>5}  ", end="")
        for name in verifier.enrolled_speakers():
            marker = "*" if name == true_name else " "
            print(f"  {name+marker:>14}", end="")
        print()

        for idx, feat in enumerate(feat_list):
            print(f"  {idx+1:>5}  ", end="")
            for name in verifier.enrolled_speakers():
                score = verifier._llr(feat, name)
                print(f"  {score:>14.3f}", end="")
            print()

    print()
    print("  * = true speaker column")
    print("  Adjust --threshold based on the score gap between true and impostors.")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Full enrollment pipeline: scan → UBM → MAP adapt → save → (calibrate)."""
    args = parse_args()

    logger.info("Loading enrollment data from: %s", SPEAKERS_DIR)
    speaker_data = collect_all_speakers(SPEAKERS_DIR)
    logger.info("Found %d speaker(s): %s", len(speaker_data), list(speaker_data.keys()))

    verifier = SpeakerVerifier()

    # ---- Train or load UBM ----
    if args.skip_ubm:
        logger.info("--skip-ubm: loading existing verifier to reuse UBM …")
        existing  = SpeakerVerifier.load()
        verifier.ubm = existing.ubm
        logger.info("UBM loaded from existing verifier.")
    else:
        all_feats = [f for feats in speaker_data.values() for f in feats]
        verifier.train_ubm(all_feats)

    # ---- Enroll each speaker ----
    logger.info("Enrolling speakers via MAP adaptation …")
    for name, feats in speaker_data.items():
        verifier.enroll(name, feats)

    # ---- Optional threshold override ----
    if args.threshold is not None:
        verifier.threshold = args.threshold
        logger.info("Decision threshold set to %.4f", verifier.threshold)

    # ---- Save ----
    verifier.save()

    # ---- Summary ----
    print()
    print("=" * 55)
    print("  Enrollment complete.")
    print(f"  Enrolled : {verifier.enrolled_speakers()}")
    print(f"  Threshold: {verifier.threshold:.4f}  (LLR >= this -> authorized)")
    print()
    print("  Tip: run with --calibrate to see LLR score distributions")
    print("       and fine-tune the threshold for your speakers.")
    print()
    print("  Next step: python src/predict.py path/to/audio.wav")
    print("=" * 55 + "\n")

    # ---- Optional calibration report ----
    if args.calibrate:
        print_calibration_report(verifier, speaker_data)


if __name__ == "__main__":
    main()
