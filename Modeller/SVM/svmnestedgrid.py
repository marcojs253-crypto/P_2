# ============================================================
# 0. IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC

import shap  # SHAP

# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH_TRAIN = "Data/TrainingData.csv"
DATA_PATH_TEST = "Data/ValidationData.csv"

TARGET_COLUMN = "target"
IRRELEVANT_COLUMNS = ["filnavn", "beta", "snr_db"]
TARGET_CLASSES = ["BlueNoise", "BrownNoise", "Clean", "PinkNoise", "VioletNoise", "WhiteNoise"]

N_SPLITS_OUTER = 5
N_SPLITS_INNER = 5
RANDOM_STATE = 42

PARAM_GRID = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"],
    "class_weight": [None, "balanced"],
}

# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data(path):
    df = pd.read_csv(path)
    for col in IRRELEVANT_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df

# ============================================================
# 3. PREPARE DATA
# ============================================================

def prepare_features(df):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    class_mapping = {name: idx for idx, name in enumerate(TARGET_CLASSES)}
    y_encoded = y.map(class_mapping).astype(int)

    label_encoder = LabelEncoder()
    label_encoder.fit(TARGET_CLASSES)

    return X, y_encoded, label_encoder

# ============================================================
# 4. SCALING
# ============================================================

def scale_features(X_train, X_val):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    return X_train_s, X_val_s, scaler

# ============================================================
# 5. NESTED CV (EVALUERING)
# ============================================================

def nested_cv_evaluation(X, y, param_grid):
    outer_cv = StratifiedKFold(n_splits=N_SPLITS_OUTER, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=N_SPLITS_INNER, shuffle=True, random_state=RANDOM_STATE)

    outer_scores = []

    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), 1):
        print(f"\n=== Outer Fold {fold} ===")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        X_train_s, X_val_s, _ = scale_features(X_train, X_val)

        grid = GridSearchCV(
            SVC(probability=True, random_state=RANDOM_STATE),
            param_grid,
            scoring="accuracy",
            cv=inner_cv,
            n_jobs=-1
        )

        grid.fit(X_train_s, y_train)

        print("Best params:", grid.best_params_)

        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_val_s)
        acc = accuracy_score(y_val, y_pred)

        print("Outer Accuracy:", acc)
        outer_scores.append(acc)

    print("\n=== Nested CV Result ===")
    print(f"Mean accuracy: {np.mean(outer_scores):.3f} ± {np.std(outer_scores):.3f}")

# ============================================================
# 6. FINAL MODEL (KORREKT)
# ============================================================

def train_final_model(X, y, param_grid):
    print("\n=== TRAINING FINAL MODEL ===")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    grid = GridSearchCV(
        SVC(probability=True, random_state=RANDOM_STATE),
        param_grid,
        scoring="accuracy",
        cv=5,
        n_jobs=-1
    )

    grid.fit(X_scaled, y)

    print("Final best params:", grid.best_params_)

    best_model = grid.best_estimator_

    return best_model, scaler

# ============================================================
# 7. EVALUATION
# ============================================================

def evaluate_model(model, scaler, X, y, label_encoder):
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    print("\n=== TEST RESULTS ===")
    print("Accuracy:", accuracy_score(y, y_pred))
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=label_encoder.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    return X_scaled, y_pred

# ============================================================
# 8. SHAP FEATURE IMPORTANCE (DIN VERSION + FIX)
# ============================================================

def shap_feature_importance(model, X_train, X_val, feature_names, save_path="Modeller/SVM/shap_feature_importance2.png"):
    print("\n=== SHAP FEATURE IMPORTANCE ===")

    X_sample = X_train[:100]

    explainer = shap.KernelExplainer(model.predict, X_sample)
    shap_values = explainer.shap_values(X_val[:50])

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    importance = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)

    plt.figure(figsize=(12,6))
    plt.bar(importance.index, importance.values)
    plt.xticks(rotation=90)
    plt.title("SHAP Feature Importance (SVM)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)

    print(f"SHAP plot saved as: {save_path}")
    plt.show()

    return importance

# ============================================================
# 9. MAIN
# ============================================================

if __name__ == "__main__":

    df_train = load_data(DATA_PATH_TRAIN)
    df_test  = load_data(DATA_PATH_TEST)

    X_train, y_train, label_encoder = prepare_features(df_train)
    X_test,  y_test,  _ = prepare_features(df_test)

    # 1. Nested CV (kun evaluering)
    nested_cv_evaluation(X_train, y_train, PARAM_GRID)

    # 2. Final model
    best_model, scaler = train_final_model(X_train, y_train, PARAM_GRID)

    # 3. Evaluation
    X_test_scaled, _ = evaluate_model(best_model, scaler, X_test, y_test, label_encoder)

    # 4. SHAP (FIXET: bruger samme scaler)
    X_train_scaled = scaler.transform(X_train)

    shap_feature_importance(
        best_model,
        X_train_scaled,
        X_test_scaled,
        X_train.columns
    )