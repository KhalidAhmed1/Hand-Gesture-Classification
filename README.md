# 🤚 Hand Gesture Classification

A machine learning pipeline that classifies **18 hand gestures** using landmark data extracted by **MediaPipe** from the **HaGRID dataset** Three models were trained and compared — Random Forest, SVM, and XGBoost — with XGBoost achieving the best performance.

---

## 📊 Results

| Model | CV Accuracy | Dev Accuracy | Test Accuracy | Macro F1 |
|---|---|---|---|---|
| Random Forest | 98.77% | 98.60% | — | 0.990 |
| SVM | 98.56% | 98.52% | — | 0.984 |
| **XGBoost** ✅ | **98.89%** | **98.64%** | **98.90%** | **0.990** |

> 🏆 **Best Model: XGBoost** with `n_estimators=363`, `max_depth=6`, `learning_rate=0.082`

---

## 🗂️ Dataset

- **Source:** HaGRID (Hand Gesture Recognition Image Dataset)
- **Landmarks:** Extracted using MediaPipe (21 keypoints × 3 coordinates)
- **Total Samples:** 25,675
- **Classes:** 18 gestures (balanced distribution)

<details>
<summary>View all 18 gestures</summary>

`call` · `dislike` · `fist` · `four` · `like` · `mute` · `ok` · `one` · `palm` · `peace` · `peace_inverted` · `rock` · `stop` · `stop_inverted` · `three` · `three2` · `two_up` · `two_up_inverted`

</details>

---

## 🔧 Pipeline

```
Raw Landmarks (21 × 3)
        │
        ▼
  Normalization
  (wrist origin + middle fingertip scale)
        │
        ▼
  Feature Engineering
  ├── Finger joint angles        (15 features)
  ├── Tip-to-wrist distances     (5 features)
  ├── Cross-tip distances        (10 features)
  ├── Thumb-specific features    (7 features)
  └── Z-axis stats per finger    (10 features)
        │
        ▼
  Final Feature Vector: 111 features
        │
        ▼
  Train / Dev / Test Split (80 / 10 / 10) — Stratified
        │
        ▼
  RandomizedSearchCV (5-fold CV)
        │
        ▼
  Random Forest │ SVM │ XGBoost
```

---

## 📁 Project Structure

```
├── Data/
│   └── hand_landmarks_data.csv
├── Hand_Gesture_Classification.ipynb
├── rf.pkl                  # Saved Random Forest model
├── svm.pkl                 # Saved SVM model
├── xgb.pkl                 # Saved XGBoost model
├── hand_landmarks.png      # Sample of each class
└── README.md
```



---

## 🧠 Model Details

### Feature Engineering
| Feature Group | Description | Count |
|---|---|---|
| Raw landmarks | Normalized x, y, z coordinates | 63 |
| Finger angles | Cosine angles between joints | 15 |
| Tip-wrist distances | Distance from each fingertip to wrist | 5 |
| Cross-tip distances | Pairwise distances between fingertips | 10 |
| Thumb features | Position & depth relative to other landmarks | 8 |
| Z-stats | Mean & std of depth per finger | 10 |
| **Total** | | **111** |

### Hyperparameter Search
All models tuned using `RandomizedSearchCV` with **20 iterations** and **5-fold cross-validation**.

| Model | Best Params |
|---|---|
| Random Forest | `max_depth=30`, `max_features=log2`, `n_estimators=154` |
| SVM | `C=30.15`, `gamma=auto`, `kernel=rbf` |
| XGBoost | `n_estimators=363`, `max_depth=6`, `learning_rate=0.082`, `subsample=0.61` |

---

## 📝 Notes

- SVM requires feature scaling (`StandardScaler`) — applied only on train set to avoid data leakage
- Most challenging gestures: **`mute`**, **`one`**, **`three`** — visually similar poses
- Real-time inference uses a **sliding window (size=10)** with majority voting for stable predictions