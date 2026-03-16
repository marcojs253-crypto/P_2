"""
Machine Learning Classification Boilerplate
------------------------------------------
Reusable template for training an XGBoost classifier with:

- Data loading
- Preprocessing
- Train/validation split
- GridSearchCV hyperparameter tuning
- Model evaluation
- Feature importance
- Optional feature selection

Author: YourName
"""

# ============================================================
# 1. IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from xgboost import XGBClassifier


# ============================================================
# 2. CONFIGURATION
# ============================================================

DATA_PATH = "path/to/your/data.csv"

TARGET_COLUMN = "target"

IRRELEVANT_COLUMNS = [
    "filnavn",
    "beta",
    "snr_db"
]

TARGET_CLASSES = [
    "BlueNoise",
    "BrownNoise",
    "Clean",
    "PinkNoise",
    "VioletNoise",
    "WhiteNoise"
]

TEST_SIZE = 0.2
RANDOM_STATE = 42


PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.03, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}


# ============================================================
# 3. LOAD DATA
# ============================================================

def load_data(path):

    df = pd.read_csv(path)

    for col in IRRELEVANT_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


# ============================================================
# 4. PREPROCESSING
# ============================================================

def prepare_features(df):

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    class_mapping = {name: idx for idx, name in enumerate(TARGET_CLASSES)}

    unknown_labels = sorted(set(y.unique()) - set(class_mapping.keys()))
    if unknown_labels:
        raise ValueError(f"Unknown classes: {unknown_labels}")

    y_encoded = y.map(class_mapping).astype(int)

    label_encoder = LabelEncoder()
    label_encoder.fit(TARGET_CLASSES)

    return X, y_encoded, label_encoder


# ============================================================
# 5. TRAIN / VALIDATION SPLIT
# ============================================================

def split_data(X, y):

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )


# ============================================================
# 6. TRAIN MODEL (GRID SEARCH)
# ============================================================

def train_model(X_train, y_train):

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(TARGET_CLASSES),
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
        tree_method="hist"
    )

    search = GridSearchCV(
        estimator=model,
        param_grid=PARAM_GRID,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    print("Best CV Score:", search.best_score_)
    print("Best Params:", search.best_params_)

    return search.best_estimator_


# ============================================================
# 7. EVALUATION
# ============================================================

def evaluate_model(model, X_val, y_val):

    y_pred = model.predict(X_val)

    print("Validation Accuracy:", accuracy_score(y_val, y_pred))

    print("\nClassification Report")
    print(classification_report(y_val, y_pred, target_names=TARGET_CLASSES))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_val, y_pred))

    return y_pred


# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================

def plot_feature_importance(model, feature_names):

    importances = model.feature_importances_

    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12,8))
    plt.title("Feature Importance (XGBoost)")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances),), feature_names[indices], rotation=90)
    plt.tight_layout()
    plt.show()

    return importances


# ============================================================
# 9. FEATURE SELECTION
# ============================================================

def remove_low_importance_features(X, importances, threshold=0.02):

    feature_names = X.columns

    features_to_keep = [
        feature
        for feature, importance in zip(feature_names, importances)
        if importance >= threshold
    ]

    X_reduced = X[features_to_keep]

    print(f"Keeping {len(features_to_keep)} / {len(feature_names)} features")

    return X_reduced