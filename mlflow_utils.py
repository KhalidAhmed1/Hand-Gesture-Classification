"""
mlflow_utils.py
---------------
MLflow utility functions for the Hand Gesture Classification experiment.
All MLflow-related logic lives here; the notebook simply imports and calls these functions.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models.signature import infer_signature

from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score
)


# ─────────────────────────────────────────────
# 1. EXPERIMENT SETUP
# ─────────────────────────────────────────────

def setup_experiment(experiment_name: str = "hand-gesture-classification") -> str:
    """
    Create (or retrieve) an MLflow experiment.
    Returns the experiment_id.
    """
    mlflow.set_tracking_uri("mlruns")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"[MLflow] Created experiment '{experiment_name}' (id={experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f"[MLflow] Using existing experiment '{experiment_name}' (id={experiment_id})")
    mlflow.set_experiment(experiment_name)
    return experiment_id


# ─────────────────────────────────────────────
# 2. DATASET LOGGING
# ─────────────────────────────────────────────

def log_dataset_info(X_train, X_dev, X_test, y_train, label_encoder, csv_path: str = None):
    """
    Log dataset statistics and optional CSV as an artifact.
    Must be called inside an active MLflow run.
    """
    mlflow.log_params({
        "dataset_train_samples"  : len(X_train),
        "dataset_dev_samples"    : len(X_dev),
        "dataset_test_samples"   : len(X_test),
        "dataset_n_features"     : X_train.shape[1],
        "dataset_n_classes"      : len(label_encoder.classes_),
        "dataset_class_names"    : json.dumps(label_encoder.classes_.tolist()),
    })

    # Class distribution on train set
    unique, counts = np.unique(y_train, return_counts=True)
    dist = {label_encoder.classes_[c]: int(counts[i]) for i, c in enumerate(unique)}
    mlflow.log_param("train_class_distribution", json.dumps(dist))

    if csv_path and os.path.exists(csv_path):
        mlflow.log_artifact(csv_path, artifact_path="dataset")
        print(f"[MLflow] Logged dataset CSV: {csv_path}")


# ─────────────────────────────────────────────
# 3. METRICS COMPUTATION & LOGGING
# ─────────────────────────────────────────────

def compute_metrics(y_true, y_pred, label_encoder):
    """
    Compute and return a metrics dict: accuracy, macro-f1, per-class f1.
    """
    acc   = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro")
    report_dict = classification_report(
        y_true, y_pred,
        target_names=label_encoder.classes_,
        output_dict=True
    )
    metrics = {
        "accuracy" : round(acc,  4),
        "macro_f1" : round(macro, 4),
    }
    for cls in label_encoder.classes_:
        metrics[f"f1_{cls}"] = round(report_dict[cls]["f1-score"], 4)
    return metrics


def log_metrics(metrics: dict, prefix: str = "dev"):
    """
    Log a dict of metrics to the active run, optionally prefixed (dev / test).
    """
    prefixed = {f"{prefix}_{k}": v for k, v in metrics.items()}
    mlflow.log_metrics(prefixed)
    print(f"[MLflow] Logged metrics ({prefix}): accuracy={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}")


# ─────────────────────────────────────────────
# 4. ARTIFACT: CONFUSION MATRIX
# ─────────────────────────────────────────────

def log_confusion_matrix(y_true, y_pred, label_encoder,
                         title: str = "Confusion Matrix",
                         artifact_path: str = "artifacts"):
    """
    Generate, save, and log a confusion-matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        ax=ax
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()

    fname = f"confusion_matrix_{title.lower().replace(' ', '_')}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    mlflow.log_artifact(fname, artifact_path=artifact_path)
    os.remove(fname)
    print(f"[MLflow] Logged confusion matrix: {fname}")


# ─────────────────────────────────────────────
# 5. ARTIFACT: CLASSIFICATION REPORT
# ─────────────────────────────────────────────

def log_classification_report(y_true, y_pred, label_encoder,
                               split: str = "dev",
                               artifact_path: str = "artifacts"):
    """
    Save and log classification report as a text file.
    """
    report = classification_report(y_true, y_pred, target_names=label_encoder.classes_)
    fname  = f"classification_report_{split}.txt"
    with open(fname, "w") as f:
        f.write(report)
    mlflow.log_artifact(fname, artifact_path=artifact_path)
    os.remove(fname)
    print(f"[MLflow] Logged classification report ({split})")


# ─────────────────────────────────────────────
# 6. MODEL LOGGING
# ─────────────────────────────────────────────

def log_sklearn_model(model, X_sample, model_name: str):
    """
    Log a scikit-learn compatible model (RF, SVM, XGBoost via sklearn API)
    with an inferred signature.
    """
    signature = infer_signature(X_sample, model.predict(X_sample))
    mlflow.sklearn.log_model(
        sk_model   = model,
        artifact_path = "model",
        signature  = signature,
        input_example = X_sample[:5],
        registered_model_name = None   # Registration done separately
    )
    print(f"[MLflow] Logged model: {model_name}")



def log_xgboost_model(model, X_sample, model_name: str):
    """
    Log an XGBoost model natively for better serialization.
    The underlying estimator is extracted from a RandomizedSearchCV wrapper if needed.
    """
    estimator = model.best_estimator_ if hasattr(model, "best_estimator_") else model
    signature = infer_signature(X_sample, estimator.predict(X_sample))
    mlflow.xgboost.log_model(
        xgb_model     = estimator,
        artifact_path = "model",
        signature     = signature,
        input_example = X_sample[:5],
        registered_model_name = None
    )
    print(f"[MLflow] Logged XGBoost model: {model_name}")


# ─────────────────────────────────────────────
# 7. HYPERPARAMETER LOGGING
# ─────────────────────────────────────────────

def log_best_params(model, prefix: str = ""):
    """
    Log hyperparameters from either:
      - a RandomizedSearchCV / GridSearchCV object  (has best_params_ & best_score_)
      - a plain sklearn / XGBoost estimator         (uses get_params())
    """
    # ── Case 1: came from a search object
    if hasattr(model, "best_params_"):
        params = {f"{prefix}{k}": v for k, v in model.best_params_.items()}
        mlflow.log_params(params)
        mlflow.log_metric(f"{prefix}best_cv_accuracy", round(model.best_score_, 4))

    # ── Case 2: plain fitted estimator
    else:
        raw    = model.get_params()
        # only log scalar / primitive values (skip nested objects)
        params = {
            f"{prefix}{k}": v for k, v in raw.items()
            if isinstance(v, (int, float, str, bool, type(None)))
        }
        mlflow.log_params(params)

    print(f"[MLflow] Logged params: {params}")


# ─────────────────────────────────────────────
# 8. FULL EXPERIMENT RUN (convenience wrapper)
# ─────────────────────────────────────────────

def run_experiment(
    run_name   : str,
    model_tag  : str,
    model,                        # plain estimator OR RandomizedSearchCV
    X_train, X_dev, X_test,
    y_train, y_dev, y_test,
    label_encoder,
    log_model_fn,                 # log_sklearn_model or log_xgboost_model
    X_dev_for_pred  = None,       # pass scaled version for SVM
    X_test_for_pred = None,
    extra_tags: dict = None,
    csv_path: str = None,
):
    """
    One-stop function: opens a run, logs everything, closes the run.
    Works with both plain fitted estimators and RandomizedSearchCV wrappers.
    Returns the run_id.
    """
    X_dev_pred  = X_dev_for_pred  if X_dev_for_pred  is not None else X_dev
    X_test_pred = X_test_for_pred if X_test_for_pred is not None else X_test

    tags = {"model_type": model_tag}
    if extra_tags:
        tags.update(extra_tags)

    with mlflow.start_run(run_name=run_name, tags=tags) as run:

        # ── Dataset
        log_dataset_info(X_train, X_dev, X_test, y_train, label_encoder, csv_path)

        # ── Hyperparameters
        log_best_params(model, prefix="param_")

        # ── Dev metrics
        y_dev_pred  = model.predict(X_dev_pred)
        dev_metrics = compute_metrics(y_dev, y_dev_pred, label_encoder)
        log_metrics(dev_metrics, prefix="dev")

        # ── Test metrics
        y_test_pred  = model.predict(X_test_pred)
        test_metrics = compute_metrics(y_test, y_test_pred, label_encoder)
        log_metrics(test_metrics, prefix="test")

        # ── Artifacts
        log_confusion_matrix(y_dev, y_dev_pred, label_encoder,
                             title=f"{model_tag} Dev Confusion Matrix")
        log_confusion_matrix(y_test, y_test_pred, label_encoder,
                             title=f"{model_tag} Test Confusion Matrix")
        log_classification_report(y_dev,  y_dev_pred,  label_encoder, split="dev")
        log_classification_report(y_test, y_test_pred, label_encoder, split="test")

        # ── Model
        log_model_fn(model, X_dev_pred, run_name)

        run_id = run.info.run_id
        print(f"[MLflow] Run '{run_name}' finished  (run_id={run_id})")

    return run_id


# ─────────────────────────────────────────────
# 9. MODEL COMPARISON CHART
# ─────────────────────────────────────────────

def log_model_comparison_chart(results: list, artifact_path: str = "comparison"):
    """
    results = [
        {"model": "Random Forest", "dev_accuracy": 0.986, "test_accuracy": 0.987,
         "dev_macro_f1": 0.986, "cv_accuracy": 0.9877},
        ...
    ]
    Logs a grouped-bar chart comparing all models.
    """
    df = pd.DataFrame(results).set_index("model")
    metrics_to_plot = ["dev_accuracy", "test_accuracy", "dev_macro_f1"]
    df = df[[c for c in metrics_to_plot if c in df.columns]]

    fig, ax = plt.subplots(figsize=(10, 6))
    x      = np.arange(len(df))
    n_bars = len(df.columns)
    width  = 0.18
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for i, (col, color) in enumerate(zip(df.columns, colors)):
        bars = ax.bar(x + i * width, df[col], width, label=col.replace("_", " ").title(), color=color)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=7.5
            )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison – Hand Gesture Classification", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (n_bars - 1) / 2)
    ax.set_xticklabels(df.index, fontsize=11)
    ax.set_ylim(0.95, 1.005)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    fname = "model_comparison.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    mlflow.log_artifact(fname, artifact_path=artifact_path)
    os.remove(fname)
    print("[MLflow] Logged model comparison chart")


# ─────────────────────────────────────────────
# 10. MODEL REGISTRY
# ─────────────────────────────────────────────

def register_best_model(run_id: str,
                        registered_name: str = "hand-gesture-xgboost-champion",
                        model_artifact_path: str = "model"):
    """
    Register the model from a given run_id into the MLflow Model Registry.
    Returns the registered model version.
    """
    model_uri = f"runs:/{run_id}/{model_artifact_path}"
    result = mlflow.register_model(model_uri=model_uri, name=registered_name)
    print(f"[MLflow] Registered model '{registered_name}' – version {result.version}")
    return result