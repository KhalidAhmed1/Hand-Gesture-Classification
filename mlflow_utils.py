"""
mlflow_utils.py
---------------
All MLflow functions for the Hand Gesture Classification experiment.
Used by Hand_Gesture_Classification.ipynb to log runs, metrics, models, and artifacts.
"""
import io
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
)


# ─────────────────────────────────────────────
#  1. Experiment Setup
# ─────────────────────────────────────────────

def setup_experiment(experiment_name: str = "Hand_Gesture_Classification") -> None:
    """Create (or reuse) an MLflow experiment with a representative name."""
    mlflow.set_experiment(experiment_name)
    print(f"[MLflow] Experiment set → '{experiment_name}'")


# ─────────────────────────────────────────────
#  2. Artifact Helpers
# ─────────────────────────────────────────────

def _log_confusion_matrix(y_true, y_pred, class_names, run_name: str) -> None:
    """Log confusion matrix directly without temporary files (Windows-safe)."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"Confusion Matrix — {run_name}", fontsize=13)
    plt.tight_layout()

    mlflow.log_figure(fig, "confusion_matrix/confusion_matrix.png")
    plt.close(fig)


def _log_classification_report(y_true, y_pred, class_names, run_name: str) -> None:
    """Log classification report directly as text artifact (no temp files)."""
    report = classification_report(y_true, y_pred, target_names=class_names)
    mlflow.log_text(
        f"Classification Report — {run_name}\n\n{report}",
        "classification_report/report.txt"
    )


def _log_dataset(dataset_path: str) -> None:
    """Log the raw dataset CSV as an MLflow artifact (if the file exists)."""
    if dataset_path and os.path.exists(dataset_path):
        mlflow.log_artifact(dataset_path, artifact_path="dataset")
    else:
        print(f"[MLflow] Dataset not found at '{dataset_path}' — skipping.")


def _log_extra_artifacts(extra_artifacts: dict) -> None:
    """
    Log extra objects (e.g. scaler) as joblib pickles — Windows-safe.
    Uses TemporaryDirectory to avoid file-lock issues on Windows.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        for name, obj in extra_artifacts.items():
            tmp_path = os.path.join(tmp_dir, f"{name}.pkl")
            joblib.dump(obj, tmp_path)
            mlflow.log_artifact(tmp_path, artifact_path="extra_artifacts")



# ─────────────────────────────────────────────
#  3. Main Logging Function
# ─────────────────────────────────────────────

def log_gesture_model(
    run_name: str,
    model,
    best_params: dict,
    cv_accuracy: float,
    dev_accuracy: float,
    y_true,
    y_pred,
    label_encoder,
    dataset_path: str = None,
    extra_artifacts: dict = None,
    register: bool = False,
    registry_name: str = None,
) -> str:
    """
    Log a single model run to MLflow.

    Parameters
    ----------
    run_name        : Descriptive name for the run (e.g. 'XGBoost_tuned').
    model           : Fitted estimator (best_estimator_ from RandomizedSearchCV).
    best_params     : Dict of best hyperparameters.
    cv_accuracy     : Cross-validation accuracy (best_score_).
    dev_accuracy    : Accuracy on the dev set.
    y_true          : True labels (dev set).
    y_pred          : Predicted labels (dev set).
    label_encoder   : Fitted LabelEncoder (to recover class names).
    dataset_path    : Path to the CSV dataset file.
    extra_artifacts : Dict of extra objects to pickle and log (e.g. scaler).
    register        : If True, register the model in the MLflow Model Registry.
    registry_name   : Name to use in the registry (required if register=True).

    Returns
    -------
    run_id : str
    """
    class_names = label_encoder.classes_

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"[MLflow] Run started → '{run_name}'  (id={run_id})")

        # ── Tags ──────────────────────────────────────────────────
        mlflow.set_tags({
            "model_type"  : type(model).__name__,
            "project"     : "Hand_Gesture_Classification",
            "dataset"     : "HaGRID_landmarks",
            "feature_set" : "landmarks+angles+distances+z_stats+thumb",
        })

        # ── Parameters ────────────────────────────────────────────
        mlflow.log_params(best_params)

        # ── Metrics ───────────────────────────────────────────────
        macro_f1    = f1_score(y_true, y_pred, average="macro")
        weighted_f1 = f1_score(y_true, y_pred, average="weighted")

        mlflow.log_metrics({
            "cv_accuracy"    : round(cv_accuracy, 6),
            "dev_accuracy"   : round(dev_accuracy, 6),
            "dev_macro_f1"   : round(macro_f1, 6),
            "dev_weighted_f1": round(weighted_f1, 6),
        })

        # ── Dataset artifact ──────────────────────────────────────
        _log_dataset(dataset_path)

        # ── Confusion matrix ──────────────────────────────────────
        _log_confusion_matrix(y_true, y_pred, class_names, run_name)

        # ── Classification report ─────────────────────────────────
        _log_classification_report(y_true, y_pred, class_names, run_name)

        # ── Extra artifacts (e.g. scaler for SVM) ─────────────────
        if extra_artifacts:
            _log_extra_artifacts(extra_artifacts)

        # ── Model ─────────────────────────────────────────────────
        model_type = type(model).__name__

        if "XGB" in model_type:
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"[MLflow] Logged → params({len(best_params)}) | "
              f"cv={cv_accuracy:.4f} | dev={dev_accuracy:.4f} | "
              f"macro_f1={macro_f1:.4f}")

        # ── Model Registry ────────────────────────────────────────
        if register:
            if not registry_name:
                registry_name = f"HandGesture_{model_type}_Champion"

            model_uri = f"runs:/{run_id}/model"
            reg = mlflow.register_model(model_uri=model_uri, name=registry_name)
            mlflow.set_tag("registry_name", registry_name)
            print(f"[MLflow] Model registered → '{registry_name}'  "
                  f"version={reg.version}")

    return run_id


# ─────────────────────────────────────────────
#  4. Promote Best Model in Registry
# ─────────────────────────────────────────────

def promote_to_production(registry_name: str, version: int) -> None:
    """
    Transition a registered model version to 'Production' stage.
    Call this after comparing all runs and settling on the best model.
    """
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name    = registry_name,
        version = version,
        stage   = "Production",
    )
    print(f"[MLflow] '{registry_name}' v{version} → Production ✓")


# ─────────────────────────────────────────────
#  5. Quick Summary Helper
# ─────────────────────────────────────────────

def print_experiment_summary(experiment_name: str = "Hand_Gesture_Classification") -> None:
    """Print a comparison table of all runs in the experiment."""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        print(f"[MLflow] No experiment found with name '{experiment_name}'")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.dev_accuracy DESC"],
    )

    print(f"\n{'─'*70}")
    print(f"  Experiment: {experiment_name}  ({len(runs)} runs)")
    print(f"{'─'*70}")
    print(f"  {'Run Name':<30} {'CV Acc':>8} {'Dev Acc':>9} {'Macro F1':>10}")
    print(f"{'─'*70}")

    for r in runs:
        m = r.data.metrics
        print(
            f"  {r.data.tags.get('mlflow.runName', r.info.run_id):<30}"
            f"  {m.get('cv_accuracy', 0):.4f}"
            f"  {m.get('dev_accuracy', 0):.4f}"
            f"  {m.get('dev_macro_f1', 0):.4f}"
        )

    print(f"{'─'*70}\n")