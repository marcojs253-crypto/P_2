# ============================================================
# SNN KLASSIFIKATIONSMODEL - STØJTYPE GENKENDELSE
# ============================================================
# Rækkefølge:
#   1.  Load og rens data  (TrainingData.csv + ValidationData.csv)
#   2.  LabelEncoder på target
#   3.  GridSearchCV med Pipeline  (5-fold CV, undgår leakage)
#   4.  Print bedste hyperparametre + CV accuracy
#   5.  Evaluér på ValidationData.csv  (alle features)
#   6.  Feature importance  (Permutation Importance på træningsdata)
#   7.  Feature selection  (85 % kumulativ importance)
#   8.  Ny GridSearchCV på udvalgte features  (re-optimerede hyperparametre)
#   9.  Sammenlign accuracy FØR / EFTER feature selection
#  10.  Valideringsanalyse: Error rate vs SNR pr. støjtype
# ============================================================

# ============================================================
# 0. IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.metrics         import (classification_report,
                                     confusion_matrix,
                                     accuracy_score,
                                     ConfusionMatrixDisplay)
from sklearn.neural_network  import MLPClassifier
from sklearn.pipeline        import Pipeline
from sklearn.inspection      import permutation_importance

# ============================================================
# 1. KONFIGURATION
# ============================================================

# Stier til datasæt (relativt til projektets rodmappe)
DATA_PATH_TRAIN = "Data/TrainingData.csv"
DATA_PATH_VAL   = "Data/ValidationData.csv"

# Navn på target-kolonnen
TARGET_COLUMN = "target"

# Kolonner der ikke er audiofeatures og skal fjernes.
# beta og snr_db har 345 manglende værdier (svarende til Clean-klassen)
IRRELEVANT_COLUMNS = ["filnavn", "beta", "snr_db"]

# Tilfældighedsfrø – bruges alle steder for reproducerbarhed
RANDOM_STATE = 42

# Andel af kumulativ feature importance der skal dækkes ved feature selection
IMPORTANCE_THRESHOLD = 0.85

# Antal gentagelser til permutation importance
# Flere gentagelser giver mere stabil ranking men tager længere tid
N_REPEATS_PERMUTATION = 10

# Hyperparameter-grid til GridSearchCV.
# Parameternavne har præfikset "clf__" fordi de tilhører Pipeline-steget "clf".
PARAM_GRID = {
    "clf__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
    "clf__activation":         ["relu", "tanh"],
    "clf__learning_rate_init": [0.001, 0.01],
    "clf__alpha":              [0.0001, 0.001],
}

# ============================================================
# STEP 1: LOAD DATA
# ============================================================

print("\n" + "="*60)
print("STEP 1: LOAD DATA")
print("="*60)

df_train = pd.read_csv(DATA_PATH_TRAIN)
df_val_raw = pd.read_csv(DATA_PATH_VAL)

# Behold en kopi af valideringsdata med metadata (fx snr_db) til analyseplots.
df_val = df_val_raw.copy()

for col in IRRELEVANT_COLUMNS:
    if col in df_train.columns:
        df_train.drop(columns=[col], inplace=True)
    if col in df_val.columns:
        df_val.drop(columns=[col], inplace=True)

print(f"Træningsdata indlæst:     {df_train.shape[0]} rækker, {df_train.shape[1]} kolonner")
print(f"Valideringsdata indlæst:  {df_val.shape[0]} rækker,  {df_val.shape[1]} kolonner")

# ============================================================
# STEP 2: LABELENCODER PÅ TARGET
# ============================================================

print("\n" + "="*60)
print("STEP 2: LABELENCODER")
print("="*60)

le      = LabelEncoder()
y_train = le.fit_transform(df_train[TARGET_COLUMN])
y_val   = le.transform(df_val[TARGET_COLUMN])

print(f"Klasser (label → indeks): { {name: idx for idx, name in enumerate(le.classes_)} }")

X_train = df_train.drop(columns=[TARGET_COLUMN])
X_val   = df_val.drop(columns=[TARGET_COLUMN])

print(f"Træningssæt:    {X_train.shape[0]} rækker, {X_train.shape[1]} features")
print(f"Valideringssæt: {X_val.shape[0]} rækker,  {X_val.shape[1]} features")

# Tjek klassebalance
class_counts    = pd.Series(y_train).value_counts().sort_index()
imbalance_ratio = class_counts.max() / class_counts.min()
print(f"Klassefordeling: { {le.classes_[i]: c for i, c in class_counts.items()} }")
if imbalance_ratio > 1.5:
    print(f"ADVARSEL: Imbalance ratio = {imbalance_ratio:.2f} – overvej class_weight='balanced'")
else:
    print(f"Datasættet er balanceret  (ratio = {imbalance_ratio:.2f})")

# ============================================================
# STEP 3: GRIDSEARCHCV MED PIPELINE
# ============================================================
# Pipeline = [StandardScaler → MLPClassifier]
# Scaleren fittes INDEN FOR hvert CV-fold → ingen data leakage
# ============================================================

print("\n" + "="*60)
print("STEP 3: GRIDSEARCHCV  (5-fold CV, Pipeline)")
print("="*60)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    MLPClassifier(max_iter=1000, random_state=RANDOM_STATE)),
])

grid_search = GridSearchCV(
    pipeline,
    PARAM_GRID,
    scoring = "accuracy",
    cv      = 5,
    n_jobs  = -1,
    verbose = 1,
)

# Fit KUN på træningsdata – valideringsdata røres ikke her
grid_search.fit(X_train, y_train)

# ============================================================
# STEP 4: BEDSTE HYPERPARAMETRE OG CV ACCURACY
# ============================================================

print("\n" + "="*60)
print("STEP 4: BEDSTE HYPERPARAMETRE")
print("="*60)

print(f"Bedste hyperparametre:  {grid_search.best_params_}")
print(f"Bedste CV accuracy:     {grid_search.best_score_:.4f}")

best_pipeline = grid_search.best_estimator_
best_scaler   = best_pipeline.named_steps["scaler"]
best_clf      = best_pipeline.named_steps["clf"]

# Skalér datasættene med scaleren der er fittet på træningsdata
X_train_scaled = best_scaler.transform(X_train)
X_val_scaled   = best_scaler.transform(X_val)

# ============================================================
# STEP 5: EVALUERING PÅ VALIDERINGSDATA  (alle features)
# ============================================================

print("\n" + "="*60)
print("STEP 5: EVALUERING PÅ VALIDERINGSDATA  (alle features)")
print("="*60)

y_pred_before = best_clf.predict(X_val_scaled)
acc_before    = accuracy_score(y_val, y_pred_before)

print(f"Accuracy (alle features): {acc_before:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred_before, target_names=le.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred_before))

# ============================================================
# STEP 6: FEATURE IMPORTANCE  (Permutation Importance)
# ============================================================
# MLPClassifier har ingen indbygget feature importance.
# Permutation Importance måler hvor meget accuracy falder når
# én feature tilfældigvis blandes – beregnet på træningsdata
# for at undgå leakage fra valideringsdata.
# ============================================================

print("\n" + "="*60)
print("STEP 6: FEATURE IMPORTANCE  (Permutation Importance)")
print("="*60)

print(f"Beregner permutation importance  ({N_REPEATS_PERMUTATION} gentagelser på træningsdata)...")

# Brug den fittede pipeline – scaler håndteres internt
perm_result = permutation_importance(
    best_pipeline,
    X_train,
    y_train,
    n_repeats    = N_REPEATS_PERMUTATION,
    random_state = RANDOM_STATE,
    n_jobs       = -1,
)

feature_names = list(X_train.columns)
importance    = pd.Series(
    perm_result.importances_mean,
    index=feature_names
).sort_values(ascending=False)

print("\nTop 10 vigtigste features (Permutation Importance):")
print(importance.head(10).to_string())

# Plot: søjlediagram over permutation importance per feature
plt.figure(figsize=(14, 6))
plt.bar(importance.index, importance.values, color="steelblue")
plt.xticks(rotation=90, fontsize=7)
plt.title("Feature Importance (SNN) – Permutation Importance per feature", fontsize=13)
plt.xlabel("Feature")
plt.ylabel("Mean accuracy drop ved permutation")
plt.tight_layout()
plt.savefig("Modeller/SNN/feature_importance.png", dpi=300)
plt.show()
print("Feature importance plot gemt: Modeller/SNN/feature_importance.png")

# ============================================================
# STEP 7: FEATURE SELECTION  (85 % kumulativ importance)
# ============================================================
# Features er sorteret efter permutation importance (faldende).
# Vi vælger de features der tilsammen udgør 85 % af den samlede
# importance.
# ValidationData.csv røres ikke her → ingen leakage.
# ============================================================

print("\n" + "="*60)
print(f"STEP 7: FEATURE SELECTION  ({int(IMPORTANCE_THRESHOLD*100)} % kumulativ importance)")
print("="*60)

sorted_features = importance.index.tolist()

# Håndter negative importance-værdier (feature bidrager ikke positivt)
# Sæt dem til 0 så kumuleringen giver mening
importance_pos = importance.clip(lower=0)

if importance_pos.sum() == 0:
    # Alle importances er 0 eller negative → behold alle features
    print("ADVARSEL: Alle permutation importances er <= 0. Beholder alle features.")
    selected_features = sorted_features
else:
    cumulative = importance_pos.cumsum() / importance_pos.sum()

    selected_features = cumulative[cumulative <= IMPORTANCE_THRESHOLD].index.tolist()

    # Tilføj den feature der krydser tærsklen
    next_idx = len(selected_features)
    if next_idx < len(importance_pos):
        selected_features.append(importance_pos.index[next_idx])

best_n = len(selected_features)

print(f"Valgte {best_n} ud af {len(sorted_features)} features")
print(f"Valgte features:\n{selected_features}")

# Plot: kumulativ importance med tærskel
if importance_pos.sum() > 0:
    plt.figure(figsize=(12, 5))
    plt.plot(range(1, len(sorted_features) + 1), cumulative.values,
             marker="o", markersize=3, linewidth=1.5, color="steelblue")
    plt.axvline(best_n, color="red", linestyle="--", label=f"Valgt: {best_n} features")
    plt.axhline(IMPORTANCE_THRESHOLD, color="gray", linestyle=":",
                label=f"Tærskel: {int(IMPORTANCE_THRESHOLD*100)} %")
    plt.xlabel("Antal features (sorteret efter permutation importance)")
    plt.ylabel("Kumulativ importance (andel)")
    plt.title(f"Kumulativ Permutation Importance – {int(IMPORTANCE_THRESHOLD*100)} % tærskel")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Modeller/SNN/feature_selection_accuracy.png", dpi=300)
    plt.show()
    print("Feature selection plot gemt: Modeller/SNN/feature_selection_accuracy.png")

# Visualisér: grøn = valgt, rød = fravalgt
colors = ["green" if f in selected_features else "red" for f in importance.index]
plt.figure(figsize=(14, 6))
plt.bar(importance.index, importance.values, color=colors)
plt.xticks(rotation=90, fontsize=7)
plt.title(f"Permutation Importance – Grøn = Valgt ({best_n} stk.), Rød = Fravalgt", fontsize=13)
plt.xlabel("Feature")
plt.ylabel("Mean accuracy drop ved permutation")
plt.tight_layout()
plt.savefig("Modeller/SNN/selected_features.png", dpi=300)
plt.show()
print("Feature selection plot gemt: Modeller/SNN/selected_features.png")

# ============================================================
# STEP 8: NY GRIDSEARCHCV PÅ UDVALGTE FEATURES
# ============================================================
# Hyperparametrene fra step 3 var optimale for alle features.
# Nu kører vi GridSearch igen på de udvalgte features, så
# hyperparametrene er optimale for det reducerede feature-sæt.
# Pipeline sikrer stadig ingen leakage i CV-foldene.
# ============================================================

print("\n" + "="*60)
print("STEP 8: NY GRIDSEARCHCV PÅ UDVALGTE FEATURES")
print("="*60)

X_train_sel = X_train[selected_features]
X_val_sel   = X_val[selected_features]

pipeline_sel = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    MLPClassifier(max_iter=1000, random_state=RANDOM_STATE)),
])

grid_search_sel = GridSearchCV(
    pipeline_sel,
    PARAM_GRID,
    scoring = "accuracy",
    cv      = 5,
    n_jobs  = -1,
    verbose = 1,
)
grid_search_sel.fit(X_train_sel, y_train)

print(f"\nBedste hyperparametre (udvalgte features): {grid_search_sel.best_params_}")
print(f"Bedste CV accuracy    (udvalgte features): {grid_search_sel.best_score_:.4f}")
print(f"\nHyperparametre step 3 ({X_train.shape[1]} features): {grid_search.best_params_}")
print(f"Hyperparametre step 8 ({best_n} features):           {grid_search_sel.best_params_}")

best_pipeline_sel = grid_search_sel.best_estimator_
scaler_sel        = best_pipeline_sel.named_steps["scaler"]
clf_sel           = best_pipeline_sel.named_steps["clf"]

X_val_sel_scaled = scaler_sel.transform(X_val_sel)
print(f"\nModel genoptrænet med {best_n} features og nye hyperparametre.")

# ============================================================
# STEP 9: SAMMENLIGNING FØR OG EFTER FEATURE SELECTION
# ============================================================

print("\n" + "="*60)
print("STEP 9: SAMMENLIGNING FØR / EFTER FEATURE SELECTION")
print("="*60)

y_pred_after = clf_sel.predict(X_val_sel_scaled)
acc_after    = accuracy_score(y_val, y_pred_after)

print(f"\n{'─'*52}")
print(f"  Accuracy FØR  feature selection:  {acc_before:.4f}  "
      f"({X_train.shape[1]} features)")
print(f"  Accuracy EFTER feature selection:  {acc_after:.4f}  "
      f"({best_n} features)")
print(f"{'─'*52}")

print("\nClassification Report  EFTER feature selection:")
print(classification_report(y_val, y_pred_after, target_names=le.classes_))

# Confusion matrix FØR og EFTER
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ConfusionMatrixDisplay.from_predictions(
    y_val, y_pred_before,
    display_labels=le.classes_,
    cmap="Blues", ax=axes[0], colorbar=False,
)
axes[0].set_title(f"Confusion Matrix  FØR  feature selection\n"
                  f"(alle {X_train.shape[1]} features,  acc={acc_before:.4f})", fontsize=11)
axes[0].tick_params(axis="x", rotation=45)

ConfusionMatrixDisplay.from_predictions(
    y_val, y_pred_after,
    display_labels=le.classes_,
    cmap="Blues", ax=axes[1], colorbar=False,
)
axes[1].set_title(f"Confusion Matrix  EFTER  feature selection\n"
                  f"({best_n} features,  acc={acc_after:.4f})", fontsize=11)
axes[1].tick_params(axis="x", rotation=45)

plt.suptitle("SNN – Confusion Matrix før og efter feature selection", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("Modeller/SNN/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()
print("Confusion matrix plot gemt: Modeller/SNN/confusion_matrix.png")

# ============================================================
# STEP 10: ERROR ANALYSIS (SNR & BETA) PÅ VALIDERINGSDATA
# ============================================================

print("\n" + "="*60)
print("STEP 10: ERROR ANALYSIS (SNR & BETA)")
print("="*60)

# Gem predictions i dataframe (samme stil som den gamle model)
df_analysis = df_val_raw.copy()
df_analysis["y_true"] = y_val
df_analysis["y_pred"] = y_pred_after
df_analysis["correct"] = df_analysis["y_true"] == df_analysis["y_pred"]

if "snr_db" not in df_analysis.columns or "beta" not in df_analysis.columns:
    print("Kan ikke lave STEP 10 fuldt ud: kolonnerne 'snr_db' og/eller 'beta' mangler i ValidationData.csv")
else:
    # Behold kun støj-rækker (Clean har typisk NaN i snr_db/beta)
    df_noise = df_analysis.dropna(subset=["snr_db", "beta"]).copy()

    if df_noise.empty:
        print("Ingen gyldige støj-rækker med både snr_db og beta. Springer STEP 10 over.")
    else:
        # ----------------------------
        # Fejlrate vs SNR
        # ----------------------------
        snr_bins = np.arange(-5, 21, 2)
        df_noise["snr_bin"] = pd.cut(df_noise["snr_db"], bins=snr_bins)

        snr_error = df_noise.groupby("snr_bin")["correct"].mean()
        snr_error = 1 - snr_error

        plt.figure(figsize=(8, 5))
        snr_error.plot(kind="bar")
        plt.title("SNN Model Error Rate vs SNR")
        plt.ylabel("Error Rate")
        plt.xlabel("SNR Bin (dB)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("Modeller/SNN/error_rate_vs_snr.png", dpi=300)
        plt.show()
        print("SNR-fejlplot gemt: Modeller/SNN/error_rate_vs_snr.png")

        # ----------------------------
        # Fejlrate vs Beta
        # ----------------------------
        beta_bins = np.linspace(-2.5, 2.5, 10)
        df_noise["beta_bin"] = pd.cut(df_noise["beta"], bins=beta_bins)

        beta_error = df_noise.groupby("beta_bin")["correct"].mean()
        beta_error = 1 - beta_error

        plt.figure(figsize=(8, 5))
        beta_error.plot(kind="bar")
        plt.title("SNN Model Error Rate vs Beta")
        plt.ylabel("Error Rate")
        plt.xlabel("Beta Bin")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("Modeller/SNN/error_rate_vs_beta.png", dpi=300)
        plt.show()
        print("Beta-fejlplot gemt: Modeller/SNN/error_rate_vs_beta.png")

        # ----------------------------
        # Eksempler på fejl
        # ----------------------------
        df_analysis["true_label"] = le.inverse_transform(df_analysis["y_true"])
        df_analysis["pred_label"] = le.inverse_transform(df_analysis["y_pred"])

        errors = df_analysis[df_analysis["correct"] == False]

        print("\nEksempler på fejl:")
        print(errors[["true_label", "pred_label", "beta", "snr_db"]].head(20))
