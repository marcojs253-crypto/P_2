#her gøres voredsd data klarrt til at blive brugt 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc


script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "df_files_2026-02-11_13-48-04.csv")
data = pd.read_csv(data_path)

# ============================================================
# 1. Data separation
# ============================================================
x_training = data[
    [
        "centroid_mean",
        "centroid_std",
        "bandwidth_mean",
        "bandwidth_std",
        "rolloff_mean",
        "rolloff_std",
        "flatness_mean",
        "flatness_std",
    ]
]

y_target = data["mos"]


# ============================================================
# 2. train test/split
# ============================================================

def evaluate_model(X, y, label):

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )


    model = MODELHER

    # Scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Cross-validation (scaled)
    X_scaled_full = scaler.fit_transform(X)
    cv_scores = cross_val_score(model, X_scaled_full, y, cv=5)
    print(f"\n{label} Cross-val accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    # Training
    model.fit(X_train_s, y_train)

    # Predictions
    y_pred = model.predict(X_test_s)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # Print results
    print(f"\n{'='*40}")
    print(f"{label} RESULTS ({model.__class__.__name__})")
    print(f"{'='*40}")
    print(f"Accuracy:      {acc:.3f}")
    print(f"Precision:     {prec:.3f}")
    print(f"Recall:        {rec:.3f}")
    print(f"ROC-AUC:       {roc_auc:.3f}")

    # Plot ROC
    plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")

