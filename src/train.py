"""
train.py

Loads the pre-computed feature CSV, performs stratified train / val / test
splits, tunes a StandardScaler + XGBClassifier pipeline with GridSearchCV,
evaluates on both held-out splits, and persists all model artifacts.

Run from the project root:
    python src/train.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    ARTIFACTS_DIR,
    CONFUSION_MATRIX_PATH,
    FEATURES_CSV,
    LABEL_ENCODER_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SPLIT,
    TRAIN_METADATA_PATH,
    VAL_SPLIT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading & label encoding
# ---------------------------------------------------------------------------


def load_data(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read the feature CSV produced by prepare_dataset.py.

    Returns:
        X: float32 array of shape (n_samples, n_features)
        y: string array of raw class labels
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Features CSV not found: {csv_path}\n"
            "Run prepare_dataset.py first to generate this file."
        )
    logger.info("Loading features from: %s", csv_path)
    df = pd.read_csv(csv_path)
    X  = df.drop(columns=["label"]).values.astype(np.float32)
    y  = df["label"].values
    logger.info("Loaded %d samples × %d features.", *X.shape)
    return X, y


def encode_labels(y_raw: np.ndarray) -> Tuple[np.ndarray, LabelEncoder]:
    """Fit a LabelEncoder on the raw string labels and return integer-encoded y."""
    le    = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    logger.info("Classes (%d): %s", len(le.classes_), list(le.classes_))
    return y_enc, le


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------


def split_data(
    X: np.ndarray, y: np.ndarray
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
]:
    """
    Stratified three-way split honouring TEST_SPLIT and VAL_SPLIT from config.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # 1. Carve out the test set
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y,
        test_size=TEST_SPLIT,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    # 2. Split the remaining pool into train and val
    # val_ratio is computed relative to the reduced pool so overall fractions hold
    val_ratio = VAL_SPLIT / (1.0 - TEST_SPLIT)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=val_ratio,
        stratify=y_tv,
        random_state=RANDOM_STATE,
    )
    logger.info(
        "Split sizes → train: %d | val: %d | test: %d",
        len(y_train), len(y_val), len(y_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------


def build_pipeline() -> Pipeline:
    """
    Return an sklearn Pipeline with two steps:
        1. StandardScaler  — zero-mean, unit-variance normalisation
        2. XGBClassifier   — gradient-boosted trees
    """
    clf = XGBClassifier(
        eval_metric="mlogloss",   # suppresses the default warning in XGB ≥ 1.6
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def _count_grid_combinations(param_grid: Dict[str, list]) -> int:
    """Return the total number of hyperparameter combinations in param_grid."""
    total = 1
    for values in param_grid.values():
        total *= len(values)
    return total


def run_grid_search(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> GridSearchCV:
    """
    Tune XGBoost hyperparameters via 3-fold stratified cross-validation.

    Scoring metric is macro-averaged F1 so all classes contribute equally,
    which matters when the 'unknown' class is present at a different frequency.
    """
    param_grid = {
        "clf__n_estimators":     [100, 200],
        "clf__max_depth":        [4, 6],
        "clf__learning_rate":    [0.05, 0.1],
        "clf__subsample":        [0.8, 1.0],
        "clf__colsample_bytree": [0.8, 1.0],
    }
    n_combos = _count_grid_combinations(param_grid)
    logger.info(
        "Starting GridSearchCV — %d combinations × 3 folds = %d fits …",
        n_combos,
        n_combos * 3,
    )

    scorer = make_scorer(f1_score, average="macro")
    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scorer,
        cv=3,
        n_jobs=-1,
        verbose=1,
        refit=True,   # refit best model on full training set after search
    )
    gs.fit(X_train, y_train)

    logger.info("Best params    : %s", gs.best_params_)
    logger.info("Best CV F1     : %.4f", gs.best_score_)
    return gs


# ---------------------------------------------------------------------------
# Evaluation & reporting
# ---------------------------------------------------------------------------


def evaluate(
    model: Any,
    X: np.ndarray,
    y_true: np.ndarray,
    le: LabelEncoder,
    split_name: str,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Generate predictions, log a classification report, and return the results.

    Returns:
        (report_dict, y_pred) for downstream use.
    """
    y_pred      = model.predict(X)
    report_str  = classification_report(y_true, y_pred, target_names=le.classes_)
    report_dict = classification_report(
        y_true, y_pred, target_names=le.classes_, output_dict=True
    )
    logger.info("\n--- %s Classification Report ---\n%s", split_name, report_str)
    return report_dict, y_pred


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    le: LabelEncoder,
    out_path: Path,
) -> None:
    """Render a colour-coded confusion matrix and write it to disk as PNG."""
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)

    fig, ax = plt.subplots(figsize=(8, 7))
    disp.plot(ax=ax, xticks_rotation="vertical", colorbar=True)
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", out_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Full training workflow: load → encode → split → tune → evaluate → save."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    X, y_raw = load_data(FEATURES_CSV)
    y, le    = encode_labels(y_raw)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # --- Train ---
    pipeline    = build_pipeline()
    grid_search = run_grid_search(pipeline, X_train, y_train)
    best_model  = grid_search.best_estimator_

    # --- Evaluate ---
    val_report,  _            = evaluate(best_model, X_val,  y_val,  le, "Validation")
    test_report, y_test_pred  = evaluate(best_model, X_test, y_test, le, "Test")

    save_confusion_matrix(y_test, y_test_pred, le, CONFUSION_MATRIX_PATH)

    # --- Persist model artifacts ---
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    logger.info("Model saved        → %s", MODEL_PATH)
    logger.info("LabelEncoder saved → %s", LABEL_ENCODER_PATH)

    # --- Persist training metadata ---
    metadata: Dict[str, Any] = {
        "best_params":      grid_search.best_params_,
        "best_cv_f1_macro": float(grid_search.best_score_),
        "val_report":       val_report,
        "test_report":      test_report,
        "train_size":       int(X_train.shape[0]),
        "val_size":         int(X_val.shape[0]),
        "test_size":        int(X_test.shape[0]),
        "n_features":       int(X.shape[1]),
        "classes":          list(le.classes_),
    }
    with open(TRAIN_METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info("Training metadata  → %s", TRAIN_METADATA_PATH)


if __name__ == "__main__":
    main()
