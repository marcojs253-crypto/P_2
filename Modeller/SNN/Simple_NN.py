import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier

# ============================================================
# 1. LOAD & PREPARE DATA
# ============================================================
df_train = pd.read_csv("https://raw.githubusercontent.com/marcojs253-crypto/P_2/refs/heads/main/Data/TrainingData.csv")

# Fjern irrelevante features
irrelevant_cols = ["filnavn", "beta", "snr_db"]
for col in irrelevant_cols:
    if col in df_train.columns:
        df_train = df_train.drop(columns=[col])

# ============================================================
# 2. FEATURES & TARGET
# ============================================================
X = df_train.drop(columns=["target"])
y = df_train["target"]

target_names = [
    "BlueNoise",
    "BrownNoise",
    "Clean",
    "PinkNoise",
    "VioletNoise",
    "WhiteNoise",
]
class_mapping = {name: idx for idx, name in enumerate(target_names)}

y_encoded = y.map(class_mapping).astype(int)

# ============================================================
# 🔴 IMPORTANT: SCALE FEATURES (VERY IMPORTANT FOR NN)
# ============================================================
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns
)

# ============================================================
# 3. TRAIN / VALIDATION SPLIT
# ============================================================
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y_encoded, test_size=0.20, stratify=y_encoded, random_state=42
)

# ============================================================
# 4. 5-FOLD CV + GRID SEARCH (NEURAL NETWORK)
# ============================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    "hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
    "activation": ["relu", "tanh"],
    "learning_rate_init": [0.001, 0.01],
    "alpha": [0.0001, 0.001],  # regularization
}

grid_combinations = (
    len(param_grid["hidden_layer_sizes"]) *
    len(param_grid["activation"]) *
    len(param_grid["learning_rate_init"]) *
    len(param_grid["alpha"])
)
print(f"Grid-kombinationer: {grid_combinations}")
print(f"Model fits i alt (kombinationer x 5 folds): {grid_combinations * 5}")

base_model = MLPClassifier(
    max_iter=1000,
    random_state=42
)

search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    verbose=0
)

search.fit(X_train, y_train)

print(f"Best CV Accuracy (5-fold): {search.best_score_:.4f}")
print(f"Best params: {search.best_params_}")

best_idx = search.best_index_
best_fold_scores = [search.cv_results_[f"split{i}_test_score"][best_idx] for i in range(cv.n_splits)]
print("Fold accuracies for bedste model:")
for i, score in enumerate(best_fold_scores, start=1):
    print(f"Fold {i}: {score:.4f}")
print(f"Mean af fold accuracies: {np.mean(best_fold_scores):.4f}")

# ============================================================
# 5. TEST ON SEPARATE DATASET
# ============================================================
model = search.best_estimator_

test_path = "https://raw.githubusercontent.com/marcojs253-crypto/P_2/refs/heads/main/Data/ValidationData.csv"
df_test = pd.read_csv(test_path)

for col in irrelevant_cols:
    if col in df_test.columns:
        df_test = df_test.drop(columns=[col])

X_test = df_test.drop(columns=["target"])
y_test = df_test["target"]

unknown_test_labels = sorted(set(y_test.unique()) - set(class_mapping.keys()))
if unknown_test_labels:
    raise ValueError(f"Ukendte klasser i test target: {unknown_test_labels}")

# Match feature order to training data before scaling.
X_test = X_test.reindex(columns=X.columns, fill_value=0)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X.columns
)
y_test_encoded = y_test.map(class_mapping).astype(int)
y_test_pred = model.predict(X_test_scaled)

print(f"\nExternal Validation Accuracy: {accuracy_score(y_test_encoded, y_test_pred):.4f}")
print("\nClassification Report (External Validation):")
print(classification_report(y_test_encoded, y_test_pred, target_names=target_names, digits=4))

print("\nConfusion Matrix (External Validation):")
print(confusion_matrix(y_test_encoded, y_test_pred))

from sklearn.inspection import permutation_importance

# ----------------------------
# 4. FEATURE IMPORTANCE (NN)
# ----------------------------

# Beregn permutation importance på validation data
result = permutation_importance(
    model,
    X_val,
    y_val,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

importances = result.importances_mean
feature_names = X.columns

# Sorter efter importance
indices = np.argsort(importances)[::-1]

# Plot
plt.figure(figsize=(12, 8))
plt.title("Feature Importance (Neural Network - Permutation)")

plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), feature_names[indices], rotation=90)

plt.tight_layout()
plt.show()

# Top 15 features
print("\nTop 15 vigtigste features:")
for i in indices[:15]:
    print(f"{feature_names[i]}: {importances[i]:.4f}")

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# ----------------------------
# 5. FEATURE SELECTION (NN)
# ----------------------------

# Baseline prediction fra den bedste model
y_val_pred = model.predict(X_val)

baseline_val_acc = accuracy_score(y_val, y_val_pred)

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=True).reset_index(drop=True)

q_values = np.linspace(0.0, 0.9, 16)

candidate_thresholds = sorted(
    set(np.quantile(importances, q_values).round(6).tolist() + [0.02])
)

threshold_results = []

for thr in candidate_thresholds:

    kept = [
        feature
        for feature, importance in zip(feature_names, importances)
        if importance >= thr
    ]

    if len(kept) < 1:
        continue

    # 🔴 IMPORTANT: rescale AFTER feature selection
    scaler_thr = StandardScaler()
    X_train_thr = X_train[kept]
    X_val_thr = X_val[kept]

    # Same best NN params from GridSearch
    thr_model = MLPClassifier(
        max_iter=1000,
        random_state=42,
        **search.best_params_
    )

    thr_model.fit(X_train_thr, y_train)

    thr_pred = thr_model.predict(X_val_thr)

    thr_acc = accuracy_score(y_val, thr_pred)

    threshold_results.append({
        "threshold": float(thr),
        "n_kept": int(len(kept)),
        "n_removed": int(len(feature_names) - len(kept)),
        "val_acc": float(thr_acc),
    })

# Safety check
if not threshold_results:
    raise ValueError("Kunne ikke evaluere nogen threshold-kandidater.")

results_df = pd.DataFrame(threshold_results)
results_df["delta_vs_baseline"] = results_df["val_acc"] - baseline_val_acc

best_row = results_df.sort_values(
    ["val_acc", "n_kept", "threshold"],
    ascending=[False, True, True]
).iloc[0]

importance_threshold = float(best_row["threshold"])

features_to_keep = [
    feature
    for feature, importance in zip(feature_names, importances)
    if importance >= importance_threshold
]

removed_features = [
    feature
    for feature, importance in zip(feature_names, importances)
    if importance < importance_threshold
]

X_reduced = X[features_to_keep]

best_acc = float(best_row["val_acc"])
delta_best = best_acc - baseline_val_acc

print(f"Keeping {len(features_to_keep)} / {len(feature_names)} features")
print(f"Removed features: {len(removed_features)}")
print(f"Optimal importance_threshold: {importance_threshold:.6f}")
print(f"Baseline Validation Accuracy: {baseline_val_acc:.4f}")
print(f"Best Validation Accuracy: {best_acc:.4f} ({delta_best:+.4f} vs baseline)")