"""
speaker_verifier.py

GMM-UBM speaker verification module.

Theory:
    A Universal Background Model (UBM) is a large GMM trained on many
    speakers that represents the "average voice population". For each
    authorized speaker we MAP-adapt the UBM means using their enrollment
    recordings — this shifts each Gaussian centroid toward that speaker's
    spectral distribution while requiring very few samples.

    At inference time the Log-Likelihood Ratio (LLR) is computed:
        LLR = log p(X | speaker_model) - log p(X | UBM)

    A positive LLR means the audio is more likely from the target speaker
    than from the general population. The decision threshold is tunable.

Reference:
    Reynolds et al. (2000) "Speaker Verification Using Adapted Gaussian
    Mixture Models", Digital Signal Processing, 10(1-3), 19-41.
"""

import copy
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import librosa
import numpy as np
from sklearn.mixture import GaussianMixture

# Entry-point scripts add src/ to sys.path before importing this module.
from config import (
    GMM_COVARIANCE_TYPE,
    MAP_RELEVANCE_FACTOR,
    N_GMM_COMPONENTS,
    N_MFCC,
    RANDOM_STATE,
    SAMPLE_RATE,
    SPEAKER_THRESHOLD,
    VERIFIER_PATH,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame-level feature extraction (different from command classifier)
# ---------------------------------------------------------------------------


def extract_speaker_features(audio: np.ndarray) -> np.ndarray:
    """
    Extract per-frame MFCC matrix suitable for GMM-based speaker modelling.

    Unlike the command classifier — which collapses frames into mean/std
    statistics — the speaker verifier needs the full sequence of frames so
    the GMM can model the distribution of spectral shapes characteristic of
    each individual's voice.

    Args:
        audio: Mono waveform array at SAMPLE_RATE.

    Returns:
        np.ndarray: shape (n_frames, N_MFCC)
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    return mfcc.T   # transpose: rows = frames, cols = coefficients


# ---------------------------------------------------------------------------
# MAP adaptation (means only)
# ---------------------------------------------------------------------------


def _map_adapt_means(
    ubm: GaussianMixture,
    features: np.ndarray,
    relevance: float,
) -> np.ndarray:
    """
    Compute MAP-adapted component means for a target speaker.

    For each component k:
        n_k      = sum of posterior responsibilities across all frames
        alpha_k  = n_k / (n_k + relevance)   [interpolation weight]
        mu_k_new = alpha_k * data_mean_k + (1 - alpha_k) * ubm_mean_k

    Components with few assigned frames (low n_k) stay close to the UBM.
    Components with many frames (high n_k) shift toward the speaker's data.

    Args:
        ubm:       Trained Universal Background Model.
        features:  Frame-level MFCC matrix, shape (n_frames, N_MFCC).
        relevance: MAP relevance factor (higher = more conservative adaptation).

    Returns:
        np.ndarray: Adapted means, shape (n_components, N_MFCC).
    """
    # Posterior assignment: P(component k | frame i)
    resp = ubm.predict_proba(features)            # (n_frames, n_components)
    n_k  = resp.sum(axis=0)                       # (n_components,)

    # Weighted data mean per component
    data_means = (resp.T @ features) / (n_k[:, np.newaxis] + 1e-10)

    # Interpolation coefficient
    alpha = n_k / (n_k + relevance)               # (n_components,)

    return alpha[:, np.newaxis] * data_means + (1.0 - alpha[:, np.newaxis]) * ubm.means_


# ---------------------------------------------------------------------------
# SpeakerVerifier
# ---------------------------------------------------------------------------


class SpeakerVerifier:
    """
    Manages the UBM and all per-speaker adapted GMMs.

    Typical workflow
    ----------------
    Training / enrollment (run once via enroll.py):
        verifier = SpeakerVerifier()
        verifier.train_ubm(all_speaker_features)
        verifier.enroll("alice", alice_features)
        verifier.enroll("bob",   bob_features)
        verifier.save()

    Inference (called from predict.py):
        verifier = SpeakerVerifier.load()
        authorized, name, score = verifier.verify(audio)
    """

    def __init__(
        self,
        n_components: int      = N_GMM_COMPONENTS,
        covariance_type: str   = GMM_COVARIANCE_TYPE,
        threshold: float       = SPEAKER_THRESHOLD,
    ) -> None:
        self.n_components    = n_components
        self.covariance_type = covariance_type
        self.threshold       = threshold
        self.ubm: Optional[GaussianMixture]       = None
        self.speaker_models: Dict[str, GaussianMixture] = {}

    # ---- UBM training ----

    def train_ubm(self, features_list: List[np.ndarray]) -> None:
        """
        Train the Universal Background Model on pooled data from all speakers.

        Using all enrolled speakers to train the UBM is a pragmatic choice
        when a large external speech corpus is unavailable. It gives each
        speaker's adapted model a meaningful contrast point.

        Args:
            features_list: One MFCC matrix per audio file (shape: n_frames × N_MFCC each).
        """
        all_feats = np.vstack(features_list)
        logger.info(
            "Training UBM — %d frames, %d files, %d components …",
            all_feats.shape[0], len(features_list), self.n_components,
        )
        self.ubm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            max_iter=200,
            n_init=3,
            random_state=RANDOM_STATE,
        )
        self.ubm.fit(all_feats)
        logger.info("UBM converged. Train log-likelihood: %.4f", self.ubm.score(all_feats))

    # ---- Speaker enrollment ----

    def enroll(self, name: str, features_list: List[np.ndarray]) -> None:
        """
        Enroll a speaker by MAP-adapting the UBM to their recordings.

        Weights and covariances are inherited from the UBM; only means are
        adapted (mean-only MAP is standard and avoids over-fitting on small
        enrollment sets).

        Args:
            name:          Speaker identifier string (e.g. "alice").
            features_list: Frame-level MFCC matrices from enrollment recordings.
        """
        if self.ubm is None:
            raise RuntimeError("Call train_ubm() before enrolling speakers.")

        speaker_feats = np.vstack(features_list)
        adapted_means = _map_adapt_means(self.ubm, speaker_feats, MAP_RELEVANCE_FACTOR)

        # Deep-copy UBM so all GMM internals (precisions_cholesky_ etc.) are intact,
        # then replace only the means.
        adapted_gmm         = copy.deepcopy(self.ubm)
        adapted_gmm.means_  = adapted_means

        self.speaker_models[name] = adapted_gmm
        logger.info(
            "Enrolled %-14s — %d files, %d frames.",
            f"'{name}'", len(features_list), speaker_feats.shape[0],
        )

    # ---- Scoring ----

    def _llr(self, features: np.ndarray, speaker_name: str) -> float:
        """
        Log-Likelihood Ratio for a single speaker.

        LLR > 0 means the audio fits the speaker model better than the UBM.
        """
        ll_spk = self.speaker_models[speaker_name].score(features)
        ll_ubm = self.ubm.score(features)
        return float(ll_spk - ll_ubm)

    def verify(self, audio: np.ndarray) -> Tuple[bool, str, float]:
        """
        Decide whether audio belongs to any enrolled authorized speaker.

        The best-matching speaker is found by maximising LLR across all
        enrolled models. If the best score exceeds self.threshold, the
        speaker is accepted.

        Args:
            audio: Mono waveform at SAMPLE_RATE.

        Returns:
            is_authorized: True if best LLR >= threshold.
            speaker_name:  Name of the closest enrolled speaker.
            best_score:    LLR value for that speaker.
        """
        if not self.speaker_models:
            raise RuntimeError("No speakers enrolled. Run enroll.py first.")

        features   = extract_speaker_features(audio)
        best_name  = ""
        best_score = -np.inf

        for name in self.speaker_models:
            score = self._llr(features, name)
            if score > best_score:
                best_score = score
                best_name  = name

        return best_score >= self.threshold, best_name, best_score

    def all_scores(self, audio: np.ndarray) -> Dict[str, float]:
        """Return LLR scores for every enrolled speaker (useful for calibration)."""
        features = extract_speaker_features(audio)
        return {name: self._llr(features, name) for name in self.speaker_models}

    def enrolled_speakers(self) -> List[str]:
        """Return sorted list of enrolled speaker names."""
        return sorted(self.speaker_models.keys())

    # ---- Persistence ----

    def save(self, path: Path = VERIFIER_PATH) -> None:
        """Serialise UBM + all speaker models to a single joblib file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("SpeakerVerifier saved → %s", path)

    @classmethod
    def load(cls, path: Path = VERIFIER_PATH) -> "SpeakerVerifier":
        """Deserialise a previously saved SpeakerVerifier from disk."""
        if not path.exists():
            raise FileNotFoundError(
                f"Speaker verifier not found: {path}\n"
                "Run enroll.py to build speaker profiles first."
            )
        instance = joblib.load(path)
        logger.info(
            "SpeakerVerifier loaded — enrolled speakers: %s",
            instance.enrolled_speakers(),
        )
        return instance
