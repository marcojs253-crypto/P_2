import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

# ============================================================
# 0. CONFIGURATION
# ============================================================
TRAIN_PATH = "https://raw.githubusercontent.com/marcojs253-crypto/P_2/refs/heads/main/Data/TrainingData.csv"
FINAL_TEST_PATH = "https://raw.githubusercontent.com/marcojs253-crypto/P_2/refs/heads/main/Data/ValidationData.csv"

TARGET_COLUMN = "target"
IRRELEVANT_COLS = ["filnavn", "beta", "snr_db"]
TARGET_NAMES = ["BlueNoise", "BrownNoise", "Clean", "PinkNoise", "VioletNoise", "WhiteNoise"]
CLASS_MAPPING = {name: idx for idx, name in enumerate(TARGET_NAMES)}

RANDOM_STATE = 42

PARAM_GRID = {
    "clf__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
    "clf__activation": ["relu", "tanh"],
    "clf__learning_rate_init": [0.001, 0.01],
    "clf__alpha": [0.0001, 0.001],
}


def load_data(path):
    df = pd.read_csv(path)
    df = df.drop(columns=[col for col in IRRELEVANT_COLS if col in df.columns])
    return df


def prepare_xy(df):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    unknown_labels = sorted(set(y.unique()) - set(CLASS_MAPPING.keys()))
    if unknown_labels:
        raise ValueError(f"Unknown classes in target: {unknown_labels}")

    y_encoded = y.map(CLASS_MAPPING).astype(int)
    return X, y_encoded


def make_mlp_pipeline(best_params=None):
    clf_kwargs = {
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
    }
    if best_params:
        for key, value in best_params.items():
            clf_kwargs[key.replace("clf__", "")] = value

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(**clf_kwargs)),
        ]
    )


def train_gridsearch(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        estimator=make_mlp_pipeline(),
        param_grid=PARAM_GRID,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
    )
    grid.fit(X, y)
    return grid


if __name__ == "__main__":
    # ============================================================
    # 1) LOAD DATA
    # ============================================================
    df_train = load_data(TRAIN_PATH)
    df_final_test = load_data(FINAL_TEST_PATH)

    X_all, y_all = prepare_xy(df_train)
    X_final_test_raw, y_final_test = prepare_xy(df_final_test)

    # Keep final test aligned with training columns.
    X_final_test_raw = X_final_test_raw.reindex(columns=X_all.columns, fill_value=0)

    # ============================================================
    # 2) SPLIT TRAINING DATA INTO TRAIN_CORE + TUNE_SET
    # ============================================================
    X_train_core, X_tune, y_train_core, y_tune = train_test_split(
        X_all,
        y_all,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )

    print(f"Train core shape: {X_train_core.shape}")
    print(f"Tune set shape:   {X_tune.shape}")

    # ============================================================
    # 3) TRAIN + TUNING ON TRAIN_CORE (CV)
    # ============================================================
    print("\n########## STEP 1: GRIDSEARCH ON TRAIN_CORE ##########")
    search = train_gridsearch(X_train_core, y_train_core)

    print(f"Best CV Accuracy: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    model_core = search.best_estimator_
    baseline_tune_pred = model_core.predict(X_tune)
    baseline_tune_acc = accuracy_score(y_tune, baseline_tune_pred)
    print(f"Tune baseline accuracy (all features): {baseline_tune_acc:.4f}")

    # ============================================================
    # 4) FEATURE IMPORTANCE + THRESHOLD SEARCH ON TUNE_SET ONLY
    # ============================================================
    print("\n########## STEP 2: THRESHOLD SEARCH ON TUNE_SET ##########")

    result = permutation_importance(
        model_core,
        X_tune,
        y_tune,
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importances = result.importances_mean
    feature_names = X_all.columns

    q_values = np.linspace(0.0, 0.9, 16)
    candidate_thresholds = sorted(
        set(np.quantile(importances, q_values).round(6).tolist() + [0.02])
    )

    threshold_results = []
    for thr in candidate_thresholds:
        kept = [f for f, imp in zip(feature_names, importances) if imp >= thr]
        if len(kept) < 1:
            continue

        X_train_thr = X_train_core[kept]
        X_tune_thr = X_tune[kept]

        thr_model = make_mlp_pipeline(best_params=search.best_params_)
        thr_model.fit(X_train_thr, y_train_core)
        thr_pred = thr_model.predict(X_tune_thr)
        thr_acc = accuracy_score(y_tune, thr_pred)

        threshold_results.append(
            {
                "threshold": float(thr),
                "n_kept": int(len(kept)),
                "n_removed": int(len(feature_names) - len(kept)),
                "tune_acc": float(thr_acc),
            }
        )

    if not threshold_results:
        raise ValueError("No threshold candidates could be evaluated.")

    results_df = pd.DataFrame(threshold_results)
    results_df["delta_vs_baseline"] = results_df["tune_acc"] - baseline_tune_acc

    best_row = results_df.sort_values(
        ["tune_acc", "n_kept", "threshold"], ascending=[False, True, True]
    ).iloc[0]

    importance_threshold = float(best_row["threshold"])
    features_to_keep = [f for f, imp in zip(feature_names, importances) if imp >= importance_threshold]
    removed_features = [f for f, imp in zip(feature_names, importances) if imp < importance_threshold]

    print(f"Keeping {len(features_to_keep)} / {len(feature_names)} features")
    print(f"Removed features: {len(removed_features)}")
    print(f"Optimal importance threshold: {importance_threshold:.6f}")
    print(f"Best tune accuracy: {best_row['tune_acc']:.4f} ({best_row['delta_vs_baseline']:+.4f} vs baseline)")

    plt.figure(figsize=(11, 5.5))
    plot_df = results_df.sort_values("threshold")
    plt.plot(plot_df["threshold"], plot_df["tune_acc"], marker="o", color="#2a9d8f", label="Tune Accuracy")
    plt.scatter([importance_threshold], [best_row["tune_acc"]], color="#e63946", s=110, zorder=5, label="Best threshold")
    plt.axvline(importance_threshold, color="#e63946", linestyle=":", alpha=0.8)
    plt.axhline(baseline_tune_acc, color="#264653", linestyle="--", label=f"Baseline={baseline_tune_acc:.4f}")
    plt.title("Tune Accuracy vs Importance Threshold")
    plt.xlabel("Importance Threshold")
    plt.ylabel("Tune Accuracy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ============================================================
    # 5) FINAL MODEL ON FULL TRAINING DATA + ONE-TIME FINAL TEST
    # ============================================================
    print("\n########## STEP 3: FINAL TRAIN + ONE-TIME FINAL TEST ##########")

    X_train_full_selected = X_all[features_to_keep]
    X_final_test_selected = X_final_test_raw[features_to_keep]

    final_model = make_mlp_pipeline(best_params=search.best_params_)
    final_model.fit(X_train_full_selected, y_all)

    y_final_pred = final_model.predict(X_final_test_selected)
    final_acc = accuracy_score(y_final_test, y_final_pred)

    print(f"\nFinal test accuracy: {final_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_final_test, y_final_pred, target_names=TARGET_NAMES))
    print("Confusion Matrix:")
    print(confusion_matrix(y_final_test, y_final_pred))