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
# 0) Configuration
# ============================================================
TRAIN_PATH = "https://raw.githubusercontent.com/marcojs253-crypto/P_2/refs/heads/main/Data/TrainingData.csv"
FINAL_TEST_PATH = "https://raw.githubusercontent.com/marcojs253-crypto/P_2/refs/heads/main/Data/ValidationData.csv"

TARGET_COLUMN = "target"
IRRELEVANT_COLS = ["filnavn", "beta", "snr_db"]
TARGET_NAMES = [
    "BlueNoise",
    "BrownNoise",
    "Clean",
    "PinkNoise",
    "VioletNoise",
    "WhiteNoise",
]
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
    # 1) Load training data + split into train_core and tune_set
    # ============================================================
    df_train = load_data(TRAIN_PATH)

    X_all, y_all = prepare_xy(df_train)

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
    # 2) Hyperparameter tuning on train_core with 5-fold CV
    # ============================================================
    search = train_gridsearch(X_train_core, y_train_core)

    print(f"Best CV Accuracy (train_core): {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    # ============================================================
    # 3) Feature threshold found ONLY on tune_set
    # ============================================================
    model_tune = search.best_estimator_
    baseline_tune_pred = model_tune.predict(X_tune)
    baseline_tune_acc = accuracy_score(y_tune, baseline_tune_pred)

    perm = permutation_importance(
        model_tune,
        X_tune,
        y_tune,
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importances = perm.importances_mean
    feature_names = X_train_core.columns

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
        raise ValueError("Could not evaluate any threshold candidates.")

    results_df = pd.DataFrame(threshold_results)
    results_df["delta_vs_baseline"] = results_df["tune_acc"] - baseline_tune_acc

    best_row = results_df.sort_values(
        ["tune_acc", "n_kept", "threshold"],
        ascending=[False, True, True],
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

    print(f"Tune baseline accuracy (all features): {baseline_tune_acc:.4f}")
    print(
        f"Best tune accuracy: {best_row['tune_acc']:.4f} "
        f"({best_row['delta_vs_baseline']:+.4f})"
    )
    print(f"Selected threshold: {importance_threshold:.6f}")
    print(f"Keeping {len(features_to_keep)} / {len(feature_names)} features")
    print(f"Removed {len(removed_features)} features")

    importance_series = pd.Series(importances, index=feature_names).sort_values(
        ascending=False
    )
    bar_colors = ["green" if f in features_to_keep else "red" for f in importance_series.index]

    plt.figure(figsize=(14, 5.5))
    plt.bar(importance_series.index, importance_series.values, color=bar_colors)
    plt.xticks(rotation=90)
    plt.title("Feature Importance (Green = Selected, Red = Dropped)")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4.8))
    plot_df = results_df.sort_values("threshold")
    plt.plot(plot_df["threshold"], plot_df["tune_acc"], marker="o", color="#2a9d8f")
    plt.axhline(
        baseline_tune_acc,
        linestyle="--",
        color="#264653",
        label=f"Baseline={baseline_tune_acc:.4f}",
    )
    plt.axvline(
        importance_threshold,
        linestyle=":",
        color="#e63946",
        label=f"Best thr={importance_threshold:.6f}",
    )
    plt.title("Tune Accuracy vs Threshold")
    plt.xlabel("importance_threshold")
    plt.ylabel("Tune Accuracy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ============================================================
    # 4) Final one-time test (only once)
    # ============================================================
    X_train_full_reduced = X_all[features_to_keep].copy()
    y_train_full = y_all.copy()

    final_model = make_mlp_pipeline(best_params=search.best_params_)
    final_model.fit(X_train_full_reduced, y_train_full)

    df_final_test = load_data(FINAL_TEST_PATH)
    X_final_test = df_final_test.drop(columns=[TARGET_COLUMN])
    y_final_test = df_final_test[TARGET_COLUMN]

    unknown_test_labels = sorted(set(y_final_test.unique()) - set(CLASS_MAPPING.keys()))
    if unknown_test_labels:
        raise ValueError(f"Unknown classes in final test target: {unknown_test_labels}")

    y_final_test_encoded = y_final_test.map(CLASS_MAPPING).astype(int)
    X_final_test = X_final_test.reindex(columns=X_all.columns, fill_value=0)
    X_final_test_reduced = X_final_test[features_to_keep].copy()

    y_final_pred = final_model.predict(X_final_test_reduced)
    final_acc = accuracy_score(y_final_test_encoded, y_final_pred)

    print("=" * 70)
    print("FINAL ONE-TIME TEST RESULT")
    print("=" * 70)
    print(f"Features used: {len(features_to_keep)}")
    print(f"Final test accuracy: {final_acc:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_final_test_encoded,
            y_final_pred,
            target_names=TARGET_NAMES,
            digits=4,
        )
    )
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_final_test_encoded, y_final_pred))
