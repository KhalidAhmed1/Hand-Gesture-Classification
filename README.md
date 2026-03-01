# 🖐️ Hand Gesture Classification

A machine learning project that classifies hand gestures in real time using **MediaPipe** landmark data extracted from the **HaGRID** dataset. Three models were trained, tracked with **MLflow**, and compared to select the best performer.

---

## 📁 Repository Structure

```
├── Hand_Gesture_Classification.ipynb        # Full notebook with MLflow 
├── mlflow_utils.py                          # All MLflow logic lives here
├── mlruns/                                  # MLflow tracking data
├── screenshots/                             # MLflow UI screenshots
├── models/
│   ├── rf_model.pkl                         # Saved Random Forest model
│   ├── svm_model.pkl                        # Saved SVM model
│   └── xgb_model.pkl                        # Saved XGBoost model
└── Data/
    └── hand_landmarks_data.csv              # Landmark dataset
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | HaGRID (Hand Gesture Recognition Image Dataset) |
| Raw features | 63 (21 keypoints × x, y, z) |
| Features after engineering | 96+ |
| Classes | 18 hand gestures |
| Class balance | ✅ Balanced |

### Gestures Covered
`call` · `dislike` · `fist` · `four` · `like` · `mute` · `ok` · `one` · `palm` · `peace` · `peace_inverted` · `rock` · `stop` · `stop_inverted` · `three` · `three2` · `two_up` · `two_up_inverted`

---

## ⚙️ Feature Engineering

Raw 3D landmarks were enriched with handcrafted geometric features:

| Feature Group | Description | Count |
|---|---|---|
| Normalized landmarks | Translated to wrist origin, scaled by middle fingertip distance | 63 |
| Finger joint angles | Cosine angles between consecutive joint triplets | 15 |
| Tip-to-wrist distances | Euclidean distance from each fingertip to wrist | 5 |
| Cross-tip distances | Pairwise distances between all fingertip combinations | 10 |
| Z-depth statistics | Mean & std of Z values per finger chain | 10 |
| Thumb features | Thumb position & depth relative to key landmarks | 8 |

---

## 🔀 Data Splits

| Split | Ratio |
|---|---|
| Train | 80% |
| Dev (Validation) | 10% |
| Test | 10% |

Stratified splitting was used to preserve class balance across all splits.

---

## 🤖 Models Trained

### 1. Random Forest
- `n_estimators=181`, `max_depth=30`, `max_features='sqrt'`, `min_samples_leaf=1`
- No feature scaling required

### 2. SVM
- Kernel: `rbf`, tuned via RandomizedSearchCV
- Required `StandardScaler` before training
- Parameters tuned: `C`, `gamma`, `kernel`

### 3. XGBoost ✅ *(Best Model)*
- `n_estimators=363`, `max_depth=6`, `learning_rate=0.082`, `subsample=0.8`, `colsample_bytree=0.7`
- No feature scaling required

---

## 📈 Model Comparison

| Model | Dev Accuracy | Test Accuracy | Dev Macro F1 |
|---|---|---|---|
| Random Forest | 98.60% | 98.71% | 0.986 |
| SVM | 98.52% | 98.63% | 0.984 |
| **XGBoost** ✅ | **98.64%** | **98.80%** | **0.987** |

> 📊 See `screenshots/` for the full MLflow comparison chart generated during the experiment.

---

## 🏆 Why XGBoost?

XGBoost was selected as the final model for the following reasons:

1. **Highest test accuracy (98.80%)** — best generalization to unseen data
2. **Highest Dev Macro F1 (0.987)** — consistent performance across all 18 classes, not just majority ones
3. **No scaling needed** — unlike SVM, XGBoost works directly on engineered features, simplifying the inference pipeline
4. **Handles feature interactions well** — the geometric features (angles, distances) benefit from sequential tree boosting
5. **Fast inference** — suitable for real-time video gesture recognition

The model was registered in the MLflow Model Registry under **`hand-gesture-xgboost-champion`**.

---

## 🧪 MLflow Experiment Tracking

All experiments were tracked using MLflow on the `research` branch.

**Experiment name:** `hand-gesture-classification`

### Runs

| Run Name | Model | Description |
|---|---|---|
| `random-forest-tuned` | Random Forest | Tuned RF with fixed best params |
| `svm-tuned` | SVM | Tuned SVM with StandardScaler |
| `xgboost-tuned` | XGBoost | Tuned XGBoost — **registered as champion** |
| `model-comparison-summary` | All | Comparison bar chart artifact |

### What Was Logged Per Run

- ✅ **Dataset** — sample counts, feature count, class names, class distribution
- ✅ **Parameters** — all model hyperparameters
- ✅ **Metrics** — accuracy & macro F1 on dev and test sets
- ✅ **Confusion matrices** — dev & test heatmaps as PNG artifacts
- ✅ **Classification reports** — per-class precision/recall/F1 as TXT artifacts
- ✅ **Model** — serialized with input signature and example

### Model Registry

| Field | Value |
|---|---|
| Name | `hand-gesture-xgboost-champion` |
| Stage | Production |
| Logged from run | `xgboost-tuned` |

---

## 🚀 Real-Time Inference

The project includes a live video inference pipeline using **OpenCV + MediaPipe**:

1. Captures webcam frames
2. Detects hand landmarks via MediaPipe
3. Applies the same normalization & feature engineering used during training
4. Predicts gesture using the XGBoost model
5. Smooths predictions over a 10-frame sliding window to reduce jitter
6. Overlays the predicted gesture label on the video feed

```python
run_gesture_recognition(model=xgb_model, label_encoder=label_encoder)
```

---

## 🛠️ Setup

```bash
pip install pandas numpy scikit-learn xgboost mediapipe opencv-python mlflow seaborn matplotlib joblib
```

---

## 🌿 Branches

| Branch | Contents |
|---|---|
| `main` | Clean notebook — no MLflow code |
| `research` | Full notebook + `mlflow_utils.py` + `mlruns/` + `screenshots/` |