# ============================================================
# 0. IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
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
TEST_SIZE = 0.2
RANDOM_STATE = 42

TARGET_CLASSES = [
    "BlueNoise", "BrownNoise", "Clean", "PinkNoise", "VioletNoise", "WhiteNoise"
]

PARAM_GRID = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"],
    "class_weight": [None, "balanced"],
    "max_iter": [500, 1000, -1]  # -1 means no limit
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
# 4. TRAIN/VALIDATION SPLIT
# ============================================================

def split_data(X, y):
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

# ============================================================
# 5. SCALE FEATURES
# ============================================================

def scale_features(X_train, X_val):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    return X_train_s, X_val_s, scaler

# ============================================================
# 6. GRID SEARCH + TRAIN MODEL
# ============================================================

def train_model_gridsearch(X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    model = SVC(class_weight="balanced", probability=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        estimator=model,
        param_grid=PARAM_GRID,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("\n=== GRID SEARCH RESULTS ===")
    print("Best CV Accuracy: {:.3f}".format(grid.best_score_))
    print("Best Parameters:", grid.best_params_)

    return grid.best_estimator_

# ============================================================
# 7. EVALUATION
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
# 8. SHAP FEATURE IMPORTANCE (med gemmefunktion)
# ============================================================

def shap_feature_importance(model, X_train, X_val, feature_names, save_path="shap_feature_importance.png"):
    """
    Beregner SHAP værdier og plotter global feature importance.
    Gemmer figuren til fil.
    """
    # Brug et mindre sample til SHAP for hastighed
    X_sample = X_train[:100]

    # KernelExplainer virker for SVM
    explainer = shap.KernelExplainer(model.predict, X_sample)

    # SHAP values for validation sample
    shap_values = explainer.shap_values(X_val[:50])

    # Global feature importance
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    importance = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)

    # Plot
    plt.figure(figsize=(12,6))
    plt.bar(importance.index, importance.values)
    plt.xticks(rotation=90)
    plt.title("SHAP Feature Importance (SVM)")
    plt.tight_layout()
    
    # Gem billedet
    plt.savefig(save_path, dpi=300)
    print(f"SHAP feature importance plot saved as: {save_path}")

    # Vis billedet
    plt.show()

    return importance

# ============================================================
# 9. MAIN SCRIPT
# ============================================================

if __name__ == "__main__":

    # --- Load data
    df_train = load_data(DATA_PATH_TRAIN)
    df_test  = load_data(DATA_PATH_TEST)

    # --- Prepare features and labels
    X_train, y_train, label_encoder = prepare_features(df_train)
    X_test,  y_test,  _            = prepare_features(df_test)

    # --- Split train/validation
    X_train, X_val, y_train, y_val = split_data(X_train, y_train)

    # --- Scale features
    X_train_s, X_val_s, scaler = scale_features(X_train, X_val)
    X_test_s = scaler.transform(X_test)

    # --- Train model with GridSearch
    model = train_model_gridsearch(X_train_s, y_train)

    # --- Evaluate on validation set
    evaluate_model(model, X_val_s, y_val, label_encoder)

    # --- SHAP feature importance (gem billedet)
    shap_importance = shap_feature_importance(
        model,
        X_train_s,
        X_val_s,
        X_train.columns,
        save_path="shap_feature_importance.png"
    )

    # --- Evaluate on test set
    print("\n=== TEST SET RESULTS ===")
    evaluate_model(model, X_test_s, y_test, label_encoder)