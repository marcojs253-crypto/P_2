#her gøres voredsd data klarrt til at blive brugt 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


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
        X, y, test_size=0.20, random_state=42
    )

    model = LinearRegression( 
        fit_intercept=True,
        copy_X=True,
        n_jobs=None,
        positive=False
        )

    # Scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Cross-validation (scaled)
    X_scaled_full = scaler.fit_transform(X)
    cv_scores = cross_val_score(model, X_scaled_full, y, cv=5)
    print(f"\n{label} Cross-val R²: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    # Training
    model.fit(X_train_s, y_train)

    # Predictions
    y_pred = model.predict(X_test_s)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print results
    print(f"\n{'='*40}")
    print(f"{label} RESULTS ({model.__class__.__name__})")
    print(f"{'='*40}")
    print(f"MAE:           {mae:.3f}")
    print(f"RMSE:          {rmse:.3f}")
    print(f"R²:            {r2:.3f}")

print("\n" + "="*70)
print("LINEAR REGRESSION")
print("="*70)
evaluate_model(x_training, y_target, "Training_data_deg")