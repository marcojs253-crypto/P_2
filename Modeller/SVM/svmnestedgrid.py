# ============================================================
# 0. IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC

import shap  # SHAP til feature importance

# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH_TRAIN = "Data/TrainingData.csv"
DATA_PATH_TEST = "Data/ValidationData.csv"

TARGET_COLUMN = "target"
IRRELEVANT_COLUMNS = ["filnavn", "beta", "snr_db"]
TARGET_CLASSES = ["BlueNoise", "BrownNoise", "Clean", "PinkNoise", "VioletNoise", "WhiteNoise"]

N_SPLITS_OUTER = 5  # Ydre CV fold
N_SPLITS_INNER = 5  # Indre CV fold for GridSearch
RANDOM_STATE = 42

PARAM_GRID = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"],
    "class_weight": [None, "balanced"],
}

# ============================================================
# 2. LOAD & CLEAN DATA
# ============================================================

def load_data(path):
    df = pd.read_csv(path)
    for col in IRRELEVANT_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df

# ============================================================
# 3. PREPARE FEATURES & LABELS
# ============================================================

def prepare_features(df):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Encode target labels
    class_mapping = {name: idx for idx, name in enumerate(TARGET_CLASSES)}
    unknown_labels = sorted(set(y.unique()) - set(class_mapping.keys()))
    if unknown_labels:
        raise ValueError(f"Unknown classes: {unknown_labels}")
    y_encoded = y.map(class_mapping).astype(int)

    label_encoder = LabelEncoder()
    label_encoder.fit(TARGET_CLASSES)

    return X, y_encoded, label_encoder

# ============================================================
# 4. SCALE FEATURES
# ============================================================

def scale_features(X_train, X_val):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    return X_train_s, X_val_s, scaler

# ============================================================
# 5. NESTED GRIDSEARCH
# ============================================================

def nested_grid_search(X, y, param_grid):
    """
    Ydre CV for performance estimation
    Indre CV (GridSearch) for hyperparameter tuning
    """
    outer_cv = StratifiedKFold(n_splits=N_SPLITS_OUTER, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=N_SPLITS_INNER, shuffle=True, random_state=RANDOM_STATE)

    outer_scores = []

    fold_num = 1
    for train_idx, val_idx in outer_cv.split(X, y):
        print(f"\n=== Outer Fold {fold_num} ===")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Scale
        X_train_s, X_val_s, scaler = scale_features(X_train, X_val)

        # Inner GridSearch
        model = SVC(max_iter=-1, probability=True, random_state=RANDOM_STATE)
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="accuracy",
            cv=inner_cv,
            n_jobs=-1
        )
        grid.fit(X_train_s, y_train)
        best_model = grid.best_estimator_

        print("Best params (inner GridSearch):", grid.best_params_)

        # Evaluer på ydre fold
        y_val_pred = best_model.predict(X_val_s)
        acc = accuracy_score(y_val, y_val_pred)
        print("Outer Fold Accuracy:", acc)
        outer_scores.append(acc)

        fold_num += 1

    mean_outer = np.mean(outer_scores)
    std_outer = np.std(outer_scores)
    print("\n=== Nested CV Results ===")
    print(f"Mean accuracy: {mean_outer:.3f} ± {std_outer:.3f}")

    # Til sidst, træner bedste model på hele dataset med bedste parametre fra sidste fold
    X_full_s, _, scaler_full = scale_features(X, X)  # fit på hele datasættet
    best_model.fit(X_full_s, y)  # træner model på hele datasættet
    return best_model, scaler_full

# ============================================================
# 6. EVALUATION
# ============================================================

def evaluate_model(model, X_val, y_val, label_encoder):
    y_pred = model.predict(X_val)
    print("\n=== EVALUATION RESULTS ===")
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    return y_pred

# ============================================================
# 7. SHAP FEATURE IMPORTANCE (med gemmefunktion)
# ============================================================

def shap_feature_importance(model, X_train, X_val, feature_names, save_path="Modeller/SVM/shap_feature_importance2.png"):
    # Brug et lille sample til SHAP for hastighed
    X_sample = X_train[:100]

    # KernelExplainer virker for SVM
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
    print(f"SHAP feature importance plot saved as: {save_path}")
    plt.show()
    return importance

# ============================================================
# 8. MAIN SCRIPT
# ============================================================

if __name__ == "__main__":

    # Load data
    df_train = load_data(DATA_PATH_TRAIN)
    df_test  = load_data(DATA_PATH_TEST)

    # Prepare features
    X_train, y_train, label_encoder = prepare_features(df_train)
    X_test,  y_test,  _            = prepare_features(df_test)

    # Nested GridSearch på træningsdata
    best_model, scaler_full = nested_grid_search(X_train, y_train, PARAM_GRID)

    # Evaluer på validation og test set
    X_val_s = scaler_full.transform(X_test)
    evaluate_model(best_model, X_val_s, y_test, label_encoder)

    # SHAP feature importance
    X_train_s, _, _ = scale_features(X_train, X_train)
    shap_feature_importance(best_model, X_train_s, X_val_s, X_train.columns)