"""
record_samples.py

Interactive CLI tool to record enrollment audio samples from a speaker
using the system microphone.

Each run collects N_ENROLLMENT_REPS clips per command and saves them as:
    data/speakers/{speaker_name}/{command}_{rep:03d}.wav

An interrupted session can be safely resumed — already-saved clips are skipped.

Run:
    python src/record_samples.py --speaker alice

Requirements (install separately if missing):
    pip install sounddevice soundfile
    # Linux only: sudo apt-get install libportaudio2
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    ENROLLMENT_COMMANDS,
    ENROLLMENT_DURATION,
    N_ENROLLMENT_REPS,
    SAMPLE_RATE,
    SPEAKERS_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------


def _require_audio_libs() -> None:
    """Raise a clear ImportError if sounddevice or soundfile are missing."""
    missing = []
    for lib in ("sounddevice", "soundfile"):
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)
    if missing:
        raise ImportError(
            f"Missing audio libraries: {', '.join(missing)}\n"
            "Install with:  pip install sounddevice soundfile\n"
            "Linux only:    sudo apt-get install libportaudio2"
        )


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def get_default_mic() -> tuple[int, str]:
    """Return (device_index, name) of the system default input device."""
    import sounddevice as sd
    device_index = sd.default.device[0]
    info = sd.query_devices(device_index)
    return device_index, info["name"]


def record_clip(duration: float, device: int) -> np.ndarray:
    """
    Record a single mono clip from the given input device.

    Args:
        duration: Recording length in seconds.
        device:   sounddevice device index to record from.

    Returns:
        np.ndarray: float32 waveform, shape (n_samples,).
    """
    import sounddevice as sd
    n_samples = int(duration * SAMPLE_RATE)
    audio = sd.rec(n_samples, samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=device)
    sd.wait()
    return audio.flatten()


def save_wav(audio: np.ndarray, path: Path) -> None:
    """Save a float32 waveform as 16-bit PCM WAV."""
    import soundfile as sf
    sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record enrollment audio for speaker verification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/record_samples.py --speaker alice\n"
            "  python src/record_samples.py --speaker bob --reps 8\n"
        ),
    )
    parser.add_argument(
        "--speaker",
        type=str,
        required=True,
        help='Speaker name/ID (e.g. "alice"). Used as the subdirectory name.',
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=N_ENROLLMENT_REPS,
        help=f"Repetitions per command (default: {N_ENROLLMENT_REPS}).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Recording session
# ---------------------------------------------------------------------------


def run_session(speaker: str, reps: int) -> None:
    """
    Guide the user through a complete enrollment recording session.

    For each command × repetition pair the function:
        1. Prints a countdown
        2. Records ENROLLMENT_DURATION seconds
        3. Saves the clip to data/speakers/{speaker}/

    Already-existing clips are skipped so an interrupted session can be resumed.
    """
    speaker_dir = SPEAKERS_DIR / speaker
    speaker_dir.mkdir(parents=True, exist_ok=True)

    total    = len(ENROLLMENT_COMMANDS) * reps
    recorded = 0

    device_index, device_name = get_default_mic()

    print()
    print("=" * 58)
    print(f"  Enrollment recording — speaker : {speaker.upper()}")
    print(f"  Microphone : [{device_index}] {device_name}")
    print(f"  Commands : {ENROLLMENT_COMMANDS}")
    print(f"  Reps/cmd : {reps}  |  Clip length : {ENROLLMENT_DURATION:.1f} s")
    print(f"  Total    : {total} clips")
    print(f"  Save dir : {speaker_dir}")
    print("=" * 58)
    print("\nPress ENTER to begin, Ctrl+C to abort at any time.\n")
    input()

    for command in ENROLLMENT_COMMANDS:
        print(f"\n--- Command: '{command.upper()}' ({reps} repetitions) ---\n")
        time.sleep(0.4)

        for rep in range(1, reps + 1):
            out_path = speaker_dir / f"{command}_{rep:03d}.wav"

            if out_path.exists():
                logger.info("Already recorded — skipping: %s", out_path.name)
                recorded += 1
                continue

            print(f"  Rep {rep}/{reps} — Prepare to say '{command}' ...")
            for tick in (3, 2, 1):
                print(f"    {tick}...")
                time.sleep(0.75)
            print("  [RECORDING]")

            audio = record_clip(ENROLLMENT_DURATION, device_index)
            save_wav(audio, out_path)

            recorded += 1
            print(f"  Saved ({recorded}/{total}): {out_path.name}\n")
            time.sleep(0.25)

    print()
    print("=" * 58)
    print(f"  Done. {recorded}/{total} clips saved for '{speaker}'.")
    print("  Next step: python src/enroll.py")
    print("=" * 58)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    _require_audio_libs()
    args = parse_args()
    try:
        run_session(args.speaker, args.reps)
    except KeyboardInterrupt:
        print("\n\nSession interrupted by user. Run again to resume.")
        sys.exit(0)


if __name__ == "__main__":
    main()
