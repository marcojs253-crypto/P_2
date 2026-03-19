import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance

# ============================================================
# 1. LOAD & PREPARE TRAINING DATA
# ============================================================
df_train = pd.read_csv("https://raw.githubusercontent.com/marcojs253-crypto/P_2/refs/heads/main/Data/TrainingData.csv")

irrelevant_cols = ["filnavn", "beta", "snr_db"]
df_train.drop(columns=[col for col in irrelevant_cols if col in df_train.columns], inplace=True)

X = df_train.drop(columns=["target"])
y = df_train["target"]

target_names = ["BlueNoise", "BrownNoise", "Clean", "PinkNoise", "VioletNoise", "WhiteNoise"]
class_mapping = {name: idx for idx, name in enumerate(target_names)}
y_encoded = y.map(class_mapping).astype(int)

# ============================================================
# 2. SCALE FEATURES
# ============================================================
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# ============================================================
# 3. TRAIN/VALIDATION SPLIT (internal for CV)
# ============================================================
# Skip train_test_split on training data
X_train = X_scaled
y_train = y_encoded


# ============================================================
# 4. GRID SEARCH + 5-FOLD CV (MLP)
# ============================================================
param_grid = {
    "hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
    "activation": ["relu", "tanh"],
    "learning_rate_init": [0.001, 0.01],
    "alpha": [0.0001, 0.001]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
base_model = MLPClassifier(max_iter=1000, random_state=42)
search = GridSearchCV(base_model, param_grid, scoring="accuracy", cv=cv, n_jobs=-1)
search.fit(X_train, y_train)

model = search.best_estimator_

print(f"Best CV Accuracy: {search.best_score_:.4f}")
print(f"Best params: {search.best_params_}")

# ============================================================
# 5. LOAD EXTERNAL VALIDATION DATASET
# ============================================================
test_path = "https://raw.githubusercontent.com/marcojs253-crypto/P_2/refs/heads/main/Data/ValidationData.csv"
df_test = pd.read_csv(test_path)
df_test.drop(columns=[col for col in irrelevant_cols if col in df_test.columns], inplace=True)

X_test = df_test.drop(columns=["target"])
y_test = df_test["target"]

# Check unknown classes
unknown_test_labels = sorted(set(y_test.unique()) - set(class_mapping.keys()))
if unknown_test_labels:
    raise ValueError(f"Unknown classes in validation set: {unknown_test_labels}")

# Align columns to training set and scale
X_test = X_test.reindex(columns=X.columns, fill_value=0)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
y_test_encoded = y_test.map(class_mapping).astype(int)

# Predict on external validation set
y_val_pred = model.predict(X_test_scaled)
baseline_val_acc = accuracy_score(y_test_encoded, y_val_pred)
print(f"\nExternal Validation Accuracy: {baseline_val_acc:.4f}")
print(classification_report(y_test_encoded, y_val_pred, target_names=target_names))
print(confusion_matrix(y_test_encoded, y_val_pred))

# ============================================================
# 6. PERMUTATION FEATURE IMPORTANCE
# ============================================================
result = permutation_importance(model, X_test_scaled, y_test_encoded, n_repeats=10, random_state=42, n_jobs=-1)
importances = result.importances_mean
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 8))
plt.title("Permutation Feature Importance (MLP)")
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
plt.tight_layout()
plt.show()

print("\nTop 15 features:")
for i in indices[:15]:
    print(f"{feature_names[i]}: {importances[i]:.4f}")

# ============================================================
# 7. FEATURE SELECTION BASED ON IMPORTANCE THRESHOLDS
# ============================================================
q_values = np.linspace(0.0, 0.9, 16)
candidate_thresholds = sorted(set(np.quantile(importances, q_values).round(6).tolist() + [0.02]))

threshold_results = []
for thr in candidate_thresholds:
    kept = [f for f, imp in zip(feature_names, importances) if imp >= thr]
    if len(kept) < 1:
        continue

    X_train_thr = X_train[kept]
    X_val_thr = X_test_scaled[kept]

    thr_model = MLPClassifier(max_iter=1000, random_state=42, **search.best_params_)
    thr_model.fit(X_train_thr, y_train)
    thr_pred = thr_model.predict(X_val_thr)
    thr_acc = accuracy_score(y_test_encoded, thr_pred)

    threshold_results.append({
        "threshold": thr,
        "n_kept": len(kept),
        "n_removed": len(feature_names) - len(kept),
        "val_acc": thr_acc
    })

results_df = pd.DataFrame(threshold_results)
results_df["delta_vs_baseline"] = results_df["val_acc"] - baseline_val_acc

best_row = results_df.sort_values(["val_acc", "n_kept", "threshold"], ascending=[False, True, True]).iloc[0]
importance_threshold = best_row["threshold"]

features_to_keep = [f for f, imp in zip(feature_names, importances) if imp >= importance_threshold]
removed_features = [f for f, imp in zip(feature_names, importances) if imp < importance_threshold]

print(f"\nKeeping {len(features_to_keep)} / {len(feature_names)} features")
print(f"Removed features: {len(removed_features)}")
print(f"Optimal importance threshold: {importance_threshold:.6f}")
print(f"Best validation accuracy: {best_row['val_acc']:.4f} ({best_row['delta_vs_baseline']:+.4f} vs baseline)")

# ============================================================
# 8. PLOT ACCURACY VS THRESHOLD
# ============================================================
plt.figure(figsize=(11, 5.5))
plt.plot(results_df["threshold"], results_df["val_acc"], marker='o', color="#2a9d8f", label="Validation Accuracy")
plt.scatter([importance_threshold], [best_row["val_acc"]], color="#e63946", s=110, zorder=5, label="Best threshold")
plt.axvline(importance_threshold, color="#e63946", linestyle=":", alpha=0.8)
plt.axhline(baseline_val_acc, color="#264653", linestyle="--", label=f"Baseline={baseline_val_acc:.4f}")
plt.title("Validation Accuracy vs Importance Threshold")
plt.xlabel("Importance Threshold")
plt.ylabel("Validation Accuracy")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()