# ============================================================
# 0. IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import GridSearchCV
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

RANDOM_STATE = 42

PARAM_GRID = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "rbf", "poly", "sigmoid"],
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
# 4. TRAIN MODEL (KUN GRIDSEARCH)
# ============================================================

def train_model_with_gridsearch(X, y, param_grid):
    print("\n=== TRAINING WITH GRIDSEARCH ===")

    scaler = StandardScaler()
    scaler.fit(X)
    joblib.dump(scaler, "svm_scaler.pkl")
    X_scaled = scaler.transform(X)
    grid = GridSearchCV(
        SVC(probability=True, random_state=RANDOM_STATE),
        param_grid,
        scoring="accuracy",
        cv=5,
        n_jobs=-1
    )

    grid.fit(X_scaled, y)

    print("Best params:", grid.best_params_)
    print("Best CV score:", grid.best_score_)

    return grid.best_estimator_, scaler

# ============================================================
# 5. EVALUATION
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
# 6. SHAP FEATURE IMPORTANCE
# ============================================================

def shap_feature_importance(model, X_train, X_val, feature_names, save_path="Modeller/SVM/shap_feature_importance3.png"):
    print("\n=== SHAP FEATURE IMPORTANCE ===")

    X_sample = X_train[:100]

    explainer = shap.KernelExplainer(model.predict_proba, X_sample)
    shap_values = explainer.shap_values(X_val[:50])

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    if len(mean_abs_shap.shape) > 1:
        mean_abs_shap = mean_abs_shap.mean(axis=1)

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
# AUTOMATISK FEATURE SELECTION BASERET PÅ SHAP
# ============================================================

def select_features_cumulative(importance, cutoff=0.90):
    """
    Vælger features, der tilsammen står for cutoff% af total SHAP importance.
    """
    # Normaliser SHAP values til sum=1
    normalized = importance / importance.sum()
    cumulative = normalized.cumsum()

    # Behold features hvor cumulative sum < cutoff
    selected = cumulative[cumulative <= cutoff].index.tolist()

    print(f"\nSelected {len(selected)} features out of {len(importance)} (covers {cutoff*100:.1f}% importance)")
    return selected

def plot_selected_features(importance, selected_features, save_path="Modeller/SVM/selected_features.png"):
    """
    Visualiserer de valgte features (grøn) vs. uvalgte (rød)
    """
    colors = ["green" if f in selected_features else "red" for f in importance.index]

    plt.figure(figsize=(12,6))
    plt.bar(importance.index, importance.values, color=colors)
    plt.xticks(rotation=90)
    plt.title("SHAP Feature Importance (Green = Selected, Red = Dropped)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

# ============================================================
# 7. MAIN
# ============================================================

if __name__ == "__main__":

    df_train = load_data(DATA_PATH_TRAIN)
    df_test  = load_data(DATA_PATH_TEST)

    X_train, y_train, label_encoder = prepare_features(df_train)
    X_test,  y_test,  _ = prepare_features(df_test)

    # 1. Train model med GridSearch
    best_model, scaler = train_model_with_gridsearch(X_train, y_train, PARAM_GRID)

    # 2. Evaluation
    X_test_scaled, _ = evaluate_model(best_model, scaler, X_test, y_test, label_encoder)

    # 3. SHAP
    X_train_scaled = scaler.transform(X_train)

    importance = shap_feature_importance(
        best_model,
        X_train_scaled,
        X_test_scaled,
        X_train.columns
    )
    # Automatisk feature selection
    selected_features = select_features_cumulative(importance, cutoff=0.90)

    # Visualiser hvilke features der blev valgt
    plot_selected_features(importance, selected_features)

    # Filtrer datasættet
    X_train_selected = X_train[selected_features]
    X_test_selected  = X_test[selected_features]

    # Train igen med de valgte features
    best_model, scaler = train_model_with_gridsearch(X_train_selected, y_train, PARAM_GRID)

    # Evaluer igen
    evaluate_model(best_model, scaler, X_test_selected, y_test, label_encoder)