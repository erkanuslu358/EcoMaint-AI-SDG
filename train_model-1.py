"""EcoMaint AI - Assignment 1.

Load the AI4I 2020 dataset, explore it, compare 3 simple ML models.
Run: python train_model.py
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")  # Save figures without opening a window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
RESULTS_DIR = ROOT / "results"
UCI_ZIP = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
SEED = 42
TARGET = "Machine failure"
FAILURE_TYPES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
SENSOR_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def get_data() -> pd.DataFrame:
    """Use a local CSV if present, otherwise download the original UCI ZIP."""
    if not DATA_PATH.is_file():
        print("Dataset is not local. Downloading it from UCI ...")
        try:
            with urlopen(UCI_ZIP, timeout=40) as response:
                zip_data = response.read()
            with ZipFile(io.BytesIO(zip_data)) as archive:
                csv_filename = next(
                    filename for filename in archive.namelist()
                    if Path(filename).name.lower() == "ai4i2020.csv"
                )
                DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
                DATA_PATH.write_bytes(archive.read(csv_filename))
        except Exception as exc:
            raise RuntimeError(
                "Dataset download failed. Download 'ai4i2020.csv' from "
                "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset "
                "and put it in the data/ folder, then rerun."
            ) from exc
    df = pd.read_csv(DATA_PATH)
    required = set(SENSOR_COLS + ["Type", TARGET] + FAILURE_TYPES)
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"The dataset is missing columns: {sorted(missing_columns)}")
    return df


def analyze_data(df: pd.DataFrame) -> None:
    """Simple EDA required by Assignment 1: missing values, stats and charts."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("\n--- DATASET CHECK ---")
    print("Rows and columns:", df.shape)
    print("Missing values:", int(df.isna().sum().sum()))
    print("Duplicate full rows:", int(df.duplicated().sum()))
    print("Failure labels:\n", df[TARGET].value_counts().sort_index().to_string())
    print("\nSensor statistics:\n", df[SENSOR_COLS].describe().round(2).to_string())

    df[SENSOR_COLS].describe().round(2).to_csv(RESULTS_DIR / "sensor_statistics.csv")
    df.isna().sum().to_csv(RESULTS_DIR / "missing_values.csv", header=["missing_count"])

    # Keep the labels from the source: one row may contain multiple failure types.
    status_counts = df[TARGET].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(["Normal", "Failure"], status_counts.values)
    ax.bar_label(bars)
    ax.set_title("Machine operating status")
    ax.set_ylabel("Number of records")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "failure_distribution.png", dpi=170)
    plt.close(fig)

    modes = df[FAILURE_TYPES].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(modes.index, modes.values)
    ax.bar_label(bars)
    ax.set_title("Failure-mode labels (can overlap)")
    ax.set_ylabel("Number of records")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "failure_modes.png", dpi=170)
    plt.close(fig)

    correlation = df["Rotational speed [rpm]"].corr(df["Torque [Nm]"])
    print(f"\nRPM / torque correlation: {correlation:.3f}")
    (RESULTS_DIR / "eda_summary.json").write_text(
        json.dumps({
            "rows": len(df),
            "columns": len(df.columns),
            "missing_values": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "normal": int(status_counts[0]),
            "failure": int(status_counts[1]),
            "failure_mode_counts": {k: int(v) for k, v in modes.items()},
            "rpm_torque_correlation": round(float(correlation), 4),
        }, indent=2), encoding="utf-8"
    )


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create physical features using only measured signals (not failure labels)."""
    x = df[["Type"] + SENSOR_COLS].copy()
    x["Temperature difference [K]"] = x["Process temperature [K]"] - x["Air temperature [K]"]
    # Convert rpm to radians per second before multiplying by torque.
    x["Mechanical power [W]"] = x["Torque [Nm]"] * x["Rotational speed [rpm]"] * 2 * np.pi / 60
    x["Torque x wear"] = x["Torque [Nm]"] * x["Tool wear [min]"]
    return x


def build_models(n_train_failures: int, n_train_normal: int):
    numeric_cols = SENSOR_COLS + [
        "Temperature difference [K]", "Mechanical power [W]", "Torque x wear"
    ]
    # OneHotEncoder is fit only on training data via Pipeline (avoids leakage).
    transform = ColumnTransformer([
        ("numbers", StandardScaler(), numeric_cols),
        ("type", OneHotEncoder(handle_unknown="ignore"), ["Type"]),
    ])
    logistic = Pipeline([
        ("preprocess", transform),
        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=SEED)),
    ])
    forest = Pipeline([
        ("preprocess", transform),
        ("classifier", RandomForestClassifier(
            n_estimators=150, class_weight="balanced", random_state=SEED, n_jobs=-1
        )),
    ])
    # XGBoost uses a class weight since failures are only ~3.4% of the observations.
    xgb = Pipeline([
        ("preprocess", transform),
        ("classifier", XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=n_train_normal / max(n_train_failures, 1),
            random_state=SEED, eval_metric="logloss", n_jobs=2,
        )),
    ])
    return {"Logistic Regression": logistic, "Random Forest": forest, "XGBoost": xgb}


def train_and_evaluate(df: pd.DataFrame) -> None:
    """Use a reproducible stratified split; report recall/precision as well as accuracy."""
    X = make_features(df)
    y = df[TARGET].astype(int)
    if y.nunique() != 2:
        raise ValueError("Dataset must contain both normal and failure records.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"\nTRAIN/TEST: {len(X_train)} / {len(X_test)} rows")
    print("Failures in test set:", int(y_test.sum()))
    models = build_models(int(y_train.sum()), int((y_train == 0).sum()))
    rows = []
    for name, model in models.items():
        print("Training:", name)
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        matrix = confusion_matrix(y_test, predicted, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        rows.append({
            "model": name,
            "accuracy": round(accuracy_score(y_test, predicted), 4),
            "precision": round(precision_score(y_test, predicted, zero_division=0), 4),
            "recall": round(recall_score(y_test, predicted, zero_division=0), 4),
            "f1": round(f1_score(y_test, predicted, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, probabilities), 4),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        })
        disp = ConfusionMatrixDisplay(matrix, display_labels=["Normal", "Failure"])
        disp.plot(values_format="d", cmap="Blues")
        plt.title(f"{name} - confusion matrix")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"confusion_{name.lower().replace(' ', '_')}.png", dpi=160)
        plt.close()

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)
    print("\n--- MEASURED TEST RESULTS ---")
    print(table.to_string(index=False))
    print(f"\nSaved charts and model_metrics.csv to: {RESULTS_DIR}")
    print("IMPORTANT: Only copy these measured values into the PDF, not unverified draft numbers.")


def main() -> None:
    df = get_data()
    analyze_data(df)
    train_and_evaluate(df)


if __name__ == "__main__":
    main()
