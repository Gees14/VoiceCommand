"""
prepare_dataset.py

Traverses the Speech Commands dataset directory, extracts a rich set of
MFCC-based acoustic features from each audio clip, and saves the result
as a flat CSV ready for sklearn pipelines.

Run from the project root:
    python src/prepare_dataset.py
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd

# Resolve sibling imports regardless of the working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    DATASET_METADATA_JSON,
    DURATION,
    FEATURES_CSV,
    INCLUDE_UNKNOWN,
    MAX_UNKNOWN_SAMPLES,
    N_MFCC,
    N_SAMPLES,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RAW_DATA_DIR,
    SAMPLE_RATE,
    TARGET_CLASSES,
    UNKNOWN_LABEL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------


def load_audio(file_path: Path) -> Optional[np.ndarray]:
    """
    Load a WAV file, resample to SAMPLE_RATE, and normalise length to N_SAMPLES.

    Clips shorter than N_SAMPLES are zero-padded at the end.
    Clips longer than N_SAMPLES are cropped from the start.

    Returns None (and logs a warning) when the file cannot be decoded.
    """
    try:
        audio, _ = librosa.load(str(file_path), sr=SAMPLE_RATE, mono=True)
        if len(audio) < N_SAMPLES:
            audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
        else:
            audio = audio[:N_SAMPLES]
        return audio
    except Exception as exc:
        logger.warning("Could not load %s — %s", file_path.name, exc)
        return None


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_features(audio: np.ndarray) -> np.ndarray:
    """
    Compute a fixed-length acoustic feature vector from a mono waveform.

    Features extracted and summarised by (mean, std) across time frames:

        • MFCC                   — N_MFCC coefficients
        • Delta-MFCC             — first-order temporal derivative of MFCC
        • Delta²-MFCC            — second-order temporal derivative of MFCC
        • Zero Crossing Rate     — rate of sign changes in the signal
        • RMS Energy             — root-mean-square frame energy
        • Spectral Centroid      — weighted mean of frequency magnitudes
        • Spectral Bandwidth     — weighted spread around the centroid
        • Spectral Rolloff       — frequency below which 85 % of energy lies

    Total vector length for N_MFCC=13:
        (13 × 3 + 5) × 2 = 88 dimensions

    Returns:
        np.ndarray: 1-D float32 feature vector.
    """
    vec: List[float] = []

    # --- MFCC family (3 matrices × N_MFCC coefs × 2 stats) ---
    mfcc    = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    d_mfcc  = librosa.feature.delta(mfcc)
    dd_mfcc = librosa.feature.delta(mfcc, order=2)

    for matrix in (mfcc, d_mfcc, dd_mfcc):
        vec.extend(np.mean(matrix, axis=1).tolist())
        vec.extend(np.std(matrix,  axis=1).tolist())

    # --- Temporal / energy features (1 value × 2 stats each) ---
    zcr = librosa.feature.zero_crossing_rate(audio)
    vec += [float(np.mean(zcr)), float(np.std(zcr))]

    rms = librosa.feature.rms(y=audio)
    vec += [float(np.mean(rms)), float(np.std(rms))]

    # --- Spectral shape features ---
    centroid  = librosa.feature.spectral_centroid(y=audio,  sr=SAMPLE_RATE)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=SAMPLE_RATE)
    rolloff   = librosa.feature.spectral_rolloff(y=audio,   sr=SAMPLE_RATE)

    for feat in (centroid, bandwidth, rolloff):
        vec += [float(np.mean(feat)), float(np.std(feat))]

    return np.array(vec, dtype=np.float32)


def _feature_column_names() -> List[str]:
    """Return the ordered column names that match the output of extract_features."""
    names: List[str] = []
    for prefix in ("mfcc", "delta_mfcc", "delta2_mfcc"):
        for stat in ("mean", "std"):
            for i in range(N_MFCC):
                names.append(f"{prefix}_{stat}_{i}")
    for feat in ("zcr", "rms", "centroid", "bandwidth", "rolloff"):
        names += [f"{feat}_mean", f"{feat}_std"]
    return names


# ---------------------------------------------------------------------------
# Dataset collection
# ---------------------------------------------------------------------------


def collect_files(
    root_dir: Path,
) -> Tuple[List[Tuple[Path, str]], List[Tuple[Path, str]]]:
    """
    Walk the dataset root and partition WAV files into target and unknown lists.

    The '_background_noise_' folder is always skipped.

    Returns:
        (target_files, unknown_files) where each entry is a (Path, label) tuple.
    """
    if not root_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {root_dir}\n"
            "Download the Speech Commands dataset and extract it to:\n"
            f"  {root_dir}"
        )

    target_files:  List[Tuple[Path, str]] = []
    unknown_files: List[Tuple[Path, str]] = []
    excluded_dirs = {"_background_noise_"}

    for class_dir in sorted(root_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name in excluded_dirs:
            continue

        label = class_dir.name
        wav_files = sorted(class_dir.glob("*.wav"))

        if label in TARGET_CLASSES:
            target_files.extend((f, label) for f in wav_files)
        elif INCLUDE_UNKNOWN:
            # Every non-target word becomes a candidate for the unknown class
            unknown_files.extend((f, UNKNOWN_LABEL) for f in wav_files)

    return target_files, unknown_files


# ---------------------------------------------------------------------------
# DataFrame construction
# ---------------------------------------------------------------------------


def build_dataframe(file_list: List[Tuple[Path, str]]) -> pd.DataFrame:
    """
    Extract features for every (path, label) pair and assemble a DataFrame.

    Progress is logged every 500 files to indicate liveness during long runs.
    Failed audio loads are skipped silently (warning already issued inside load_audio).
    """
    col_names = _feature_column_names() + ["label"]
    records: List[tuple] = []
    total = len(file_list)

    for idx, (file_path, label) in enumerate(file_list):
        if idx % 500 == 0:
            logger.info("  Processing: %d / %d files …", idx, total)

        audio = load_audio(file_path)
        if audio is None:
            continue

        feat = extract_features(audio)
        records.append((*feat.tolist(), label))

    logger.info("  Processing: %d / %d files — complete.", total, total)

    if not records:
        raise RuntimeError("No audio files could be loaded. Check dataset path and format.")

    return pd.DataFrame(records, columns=col_names)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Orchestrate full dataset preparation: scan → extract → save."""
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Scanning dataset at: %s", RAW_DATA_DIR)
    target_files, unknown_files = collect_files(RAW_DATA_DIR)
    logger.info(
        "Target samples   : %d  (%d classes: %s)",
        len(target_files),
        len(TARGET_CLASSES),
        TARGET_CLASSES,
    )
    logger.info("Candidate unknown : %d", len(unknown_files))

    # Subsample unknown class to avoid imbalance
    if INCLUDE_UNKNOWN and unknown_files:
        random.shuffle(unknown_files)
        unknown_files = unknown_files[:MAX_UNKNOWN_SAMPLES]
        logger.info("Unknown (capped)  : %d (max=%d)", len(unknown_files), MAX_UNKNOWN_SAMPLES)

    all_files = target_files + (unknown_files if INCLUDE_UNKNOWN else [])
    random.shuffle(all_files)

    logger.info(
        "Extracting features for %d audio files — this may take several minutes …",
        len(all_files),
    )
    df = build_dataframe(all_files)

    # ---- Report ----
    logger.info("Dataset shape      : %s", df.shape)
    logger.info("Class distribution :")
    dist = df["label"].value_counts()
    for cls, count in dist.items():
        logger.info("  %-14s : %d", cls, count)

    # ---- Persist ----
    df.to_csv(FEATURES_CSV, index=False)
    logger.info("Features CSV saved → %s", FEATURES_CSV)

    metadata = {
        "dataset_shape":      list(df.shape),
        "n_features":         int(df.shape[1] - 1),
        "classes":            sorted(df["label"].unique().tolist()),
        "class_distribution": {k: int(v) for k, v in dist.items()},
        "sample_rate":        SAMPLE_RATE,
        "duration_sec":       DURATION,
        "n_mfcc":             N_MFCC,
        "include_unknown":    INCLUDE_UNKNOWN,
        "max_unknown_samples": MAX_UNKNOWN_SAMPLES,
    }
    with open(DATASET_METADATA_JSON, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info("Metadata saved     → %s", DATASET_METADATA_JSON)


if __name__ == "__main__":
    main()
