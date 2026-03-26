# ============================================================
# SVM KLASSIFIKATIONSMODEL - STØJTYPE GENKENDELSE
# ============================================================
# Rækkefølge:
#   1.  Load og rens data  (TrainingData.csv + ValidationData.csv)
#   2.  LabelEncoder på target
#   3.  GridSearchCV med Pipeline  (5-fold CV, undgår leakage)
#   4.  Print bedste hyperparametre + CV accuracy
#   5.  Evaluér på ValidationData.csv  (alle features)
#   6.  SHAP feature importance på træningsdata
#   7.  Feature selection  (90 % kumulativ SHAP importance)
#   8.  Genoptræn SVM på udvalgte features  (samme hyperparametre)
#   9.  Sammenlign accuracy FØR / EFTER feature selection
#  10.  Error analysis (SNR & BETA) på valideringsdata
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
                                     accuracy_score)
from sklearn.svm             import SVC
from sklearn.pipeline        import Pipeline

import shap

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

# Hyperparameter-grid til GridSearchCV.
# Parameternavne har præfikset "svc__" fordi de tilhører Pipeline-steget "svc".
# class_weight=None: datasættet er balanceret, så "balanced" er redundant.
PARAM_GRID = {
    "svc__C":            [0.1, 1, 10, 100],
    "svc__kernel":       ["linear", "rbf", "poly", "sigmoid"],
    "svc__gamma":        ["scale", "auto"],
    "svc__class_weight": [None],
}

# ============================================================
# STEP 1: LOAD DATA
# ============================================================

print("\n" + "="*60)
print("STEP 1: LOAD DATA")
print("="*60)

# Indlæs træningsdata og valideringsdata fra CSV
df_train = pd.read_csv(DATA_PATH_TRAIN)
df_val_raw = pd.read_csv(DATA_PATH_VAL)

# Behold en rå kopi af valideringsdata med metadata (snr_db, beta) til step 10.
df_val = df_val_raw.copy()

# Fjern irrelevante kolonner fra begge datasæt
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

# Fit LabelEncoder på træningsdata og transformer begge datasæt
le      = LabelEncoder()
y_train = le.fit_transform(df_train[TARGET_COLUMN])
y_val   = le.transform(df_val[TARGET_COLUMN])

print(f"Klasser (label → indeks): { {name: idx for idx, name in enumerate(le.classes_)} }")

# Features: alle kolonner undtagen target
X_train = df_train.drop(columns=[TARGET_COLUMN])
X_val   = df_val.drop(columns=[TARGET_COLUMN])

print(f"Træningssæt:    {X_train.shape[0]} rækker, {X_train.shape[1]} features")
print(f"Valideringssæt: {X_val.shape[0]} rækker,  {X_val.shape[1]} features")

# ============================================================
# STEP 3: GRIDSEARCHCV MED PIPELINE
# ============================================================
# Pipeline = [StandardScaler → SVC]
# Scaleren fitttes INDEN FOR hvert CV-fold → ingen data leakage
# ============================================================

print("\n" + "="*60)
print("STEP 3: GRIDSEARCHCV  (5-fold CV, Pipeline)")
print("="*60)

# Byg pipeline: scaler-steget hedder "scaler", SVM-steget hedder "svc"
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svc",    SVC(probability=True, random_state=RANDOM_STATE)),
])

# GridSearchCV med 5-fold cross-validation på træningsdata
grid_search = GridSearchCV(
    pipeline,
    PARAM_GRID,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
    verbose=1,
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

# Udtræk den fittede Pipeline og dens individuelle komponenter
best_pipeline = grid_search.best_estimator_
best_scaler   = best_pipeline.named_steps["scaler"]   # Fitted StandardScaler
best_svc      = best_pipeline.named_steps["svc"]      # Fitted SVC
best_kernel   = grid_search.best_params_["svc__kernel"]

# Skalér datasættene med scaleren der er fittet på træningsdata
X_train_scaled = best_scaler.transform(X_train)
X_val_scaled   = best_scaler.transform(X_val)    # Transform – ingen leakage

# ============================================================
# STEP 5: EVALUERING PÅ VALIDERINGSDATA  (alle features)
# ============================================================

print("\n" + "="*60)
print("STEP 5: EVALUERING PÅ VALIDERINGSDATA  (alle features)")
print("="*60)

y_pred_before = best_svc.predict(X_val_scaled)
acc_before    = accuracy_score(y_val, y_pred_before)

print(f"Accuracy (alle features): {acc_before:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred_before, target_names=le.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred_before))

# ============================================================
# STEP 6: SHAP FEATURE IMPORTANCE PÅ TRÆNINGSDATA
# ============================================================
# - Lineær kernel  → shap.LinearExplainer  (hurtig, eksakt)
# - Andre kernels  → shap.KernelExplainer  (model-agnostisk, langsommere)
# SHAP beregnes på skalerede træningsdata (ingen leakage)
# ============================================================

print("\n" + "="*60)
print("STEP 6: SHAP FEATURE IMPORTANCE")
print("="*60)

feature_names = list(X_train.columns)

if best_kernel == "linear":
    # LinearExplainer virker direkte med lineær SVC (har coef_-attribut)
    print("Bruger shap.LinearExplainer  (lineær kernel)")
    explainer   = shap.LinearExplainer(best_svc, X_train_scaled)
    shap_values = explainer.shap_values(X_train_scaled)

else:
    # KernelExplainer er model-agnostisk men langsommere
    print(f"Bruger shap.KernelExplainer  (kernel: {best_kernel})")

    # Brug 100 baggrundsprøver for at holde beregningen overkommelig
    background = shap.sample(
        pd.DataFrame(X_train_scaled, columns=feature_names),
        100,
        random_state=RANDOM_STATE,
    )
    explainer = shap.KernelExplainer(best_svc.predict_proba, background)

    # Beregn SHAP på de første 300 træningsprøver (KernelExplainer er langsom)
    shap_values = explainer.shap_values(X_train_scaled[:300])

# Håndtér multiclass output fra SHAP:
# - Liste af arrays → én per klasse (KernelExplainer)
# - 3D array (n_samples, n_features, n_classifiers) → LinearExplainer med one-vs-one
#   (6 klasser giver C(6,2) = 15 binære klassifikatorer)
if isinstance(shap_values, list):
    # Tag gennemsnit af mean |SHAP| på tværs af klasser
    mean_abs_shap = np.mean(
        [np.abs(sv).mean(axis=0) for sv in shap_values],
        axis=0
    )
else:
    # Tag gennemsnit over samples (axis=0)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

# Hvis resultatet stadig er 2D (fx (43, 15)), tag gennemsnit over den resterende akse
if mean_abs_shap.ndim > 1:
    mean_abs_shap = mean_abs_shap.mean(axis=-1)   # → (n_features,)

# Sortér features efter importance (faldende)
importance = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)

print("\nTop 10 vigtigste features (SHAP):")
print(importance.head(10).to_string())

# Plot: søjlediagram over mean |SHAP| per feature
plt.figure(figsize=(14, 6))
plt.bar(importance.index, importance.values, color="steelblue")
plt.xticks(rotation=90, fontsize=7)
plt.title("SHAP Feature Importance (SVM) – Mean |SHAP| per feature", fontsize=13)
plt.xlabel("Feature")
plt.ylabel("Mean |SHAP value|")
plt.tight_layout()
plt.savefig("Modeller/SVM/shap_feature_importance.png", dpi=300)
plt.show()
print("SHAP plot gemt: Modeller/SVM/shap_feature_importance.png")

# ============================================================
# STEP 7: FEATURE SELECTION  (90 % KUMULATIV SHAP IMPORTANCE)
# ============================================================

print("\n" + "="*60)
print("STEP 7: FEATURE SELECTION  (90 % kumulativ SHAP importance)")
print("="*60)

# Normalisér importance-værdierne så de summer til 1
normalized = importance / importance.sum()

# Beregn kumulativ sum (features er allerede sorteret faldende)
cumulative = normalized.cumsum()

# Vælg features indtil den kumulative sum passerer 90 %
selected_features = cumulative[cumulative <= 0.90].index.tolist()

# Tilføj den næste feature for at sikre vi rammer mindst 90 %
remaining = [f for f in cumulative.index if f not in selected_features]
if remaining:
    selected_features.append(remaining[0])

print(f"Valgte {len(selected_features)} ud af {len(importance)} features  "
      f"(dækning: {cumulative[selected_features[-1]]:.2%})")
print(f"Valgte features:\n{selected_features}")

# Visualisér: grøn = valgt, rød = fravalgt
colors = ["green" if f in selected_features else "red" for f in importance.index]
plt.figure(figsize=(14, 6))
plt.bar(importance.index, importance.values, color=colors)
plt.xticks(rotation=90, fontsize=7)
plt.title("SHAP Feature Importance – Grøn = Valgt, Rød = Fravalgt  (90 % tærskel)", fontsize=13)
plt.xlabel("Feature")
plt.ylabel("Mean |SHAP value|")
plt.tight_layout()
plt.savefig("Modeller/SVM/selected_features.png", dpi=300)
plt.show()
print("Feature selection plot gemt: Modeller/SVM/selected_features.png")

# ============================================================
# STEP 8: GENOPTRÆN SVM PÅ UDVALGTE FEATURES
# ============================================================
# Bruger de præcis samme hyperparametre som fundet i GridSearch.
# Ny scaler fittets KUN på de udvalgte træningsfeatures → ingen leakage.
# ============================================================

print("\n" + "="*60)
print("STEP 8: GENOPTRÆNING PÅ UDVALGTE FEATURES")
print("="*60)

# Reducer datasæt til kun de valgte features
X_train_sel = X_train[selected_features]
X_val_sel   = X_val[selected_features]

# Rens parameternavne: fjern "svc__"-præfikset fra GridSearch-resultaterne
best_params_clean = {
    key.replace("svc__", ""): val
    for key, val in grid_search.best_params_.items()
}
print(f"Genbrugte hyperparametre: {best_params_clean}")

# Fit en ny StandardScaler KUN på de valgte træningsfeatures
scaler_sel         = StandardScaler()
X_train_sel_scaled = scaler_sel.fit_transform(X_train_sel)
X_val_sel_scaled   = scaler_sel.transform(X_val_sel)    # Transform – ingen leakage

# Opbyg og træn en ny SVC med de bedste hyperparametre
svc_sel = SVC(
    probability=True,
    random_state=RANDOM_STATE,
    **best_params_clean,
)
svc_sel.fit(X_train_sel_scaled, y_train)
print(f"Model genoptrænet med {len(selected_features)} features.")

# ============================================================
# STEP 9: SAMMENLIGNING FØR OG EFTER FEATURE SELECTION
# ============================================================

print("\n" + "="*60)
print("STEP 9: SAMMENLIGNING FØR / EFTER FEATURE SELECTION")
print("="*60)

# Forudsig på valideringsdata med den reducerede model
y_pred_after = svc_sel.predict(X_val_sel_scaled)
acc_after    = accuracy_score(y_val, y_pred_after)

# Oversigt
print(f"\n{'─'*52}")
print(f"  Accuracy FØR  feature selection:  {acc_before:.4f}  "
      f"({X_train.shape[1]} features)")
print(f"  Accuracy EFTER feature selection:  {acc_after:.4f}  "
      f"({len(selected_features)} features)")
print(f"{'─'*52}")

# Confusion matrix FØR
print("\nConfusion Matrix  FØR  feature selection:")
print(confusion_matrix(y_val, y_pred_before))

# Confusion matrix EFTER
print("\nConfusion Matrix  EFTER feature selection:")
print(confusion_matrix(y_val, y_pred_after))

# Classification report EFTER
print("\nClassification Report  EFTER feature selection:")
print(classification_report(y_val, y_pred_after, target_names=le.classes_))

# ============================================================
# STEP 10: ERROR ANALYSIS (SNR & BETA)
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
        df_noise["snr_int"] = pd.to_numeric(df_noise["snr_db"], errors="coerce").round().astype("Int64")
        df_noise = df_noise.dropna(subset=["snr_int"]).copy()

        snr_error = df_noise.groupby("snr_int")["correct"].mean().sort_index()
        snr_error = 1 - snr_error

        min_snr = int(snr_error.index.min())
        max_snr = int(snr_error.index.max())
        full_snr_range = np.arange(min_snr, max_snr + 1, 1)
        snr_error = snr_error.reindex(full_snr_range)

        plt.figure(figsize=(8, 5))
        snr_error.plot(kind="bar")
        plt.title("SVM Model Error Rate vs SNR")
        plt.ylabel("Error Rate")
        plt.xlabel("SNR (dB)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("Modeller/SVM/error_rate_vs_snr.png", dpi=300)
        plt.show()
        print("SNR-fejlplot gemt: Modeller/SVM/error_rate_vs_snr.png")

        # ----------------------------
        # Fejlrate vs Beta
        # ----------------------------
        beta_bins = np.linspace(-2.5, 2.5, 10)
        df_noise["beta_bin"] = pd.cut(df_noise["beta"], bins=beta_bins)

        beta_error = df_noise.groupby("beta_bin")["correct"].mean()
        beta_error = 1 - beta_error

        plt.figure(figsize=(8, 5))
        beta_error.plot(kind="bar")
        plt.title("SVM Model Error Rate vs Beta")
        plt.ylabel("Error Rate")
        plt.xlabel("Beta Bin")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("Modeller/SVM/error_rate_vs_beta.png", dpi=300)
        plt.show()
        print("Beta-fejlplot gemt: Modeller/SVM/error_rate_vs_beta.png")

        # ----------------------------
        # Eksempler på fejl
        # ----------------------------
        df_analysis["true_label"] = le.inverse_transform(df_analysis["y_true"])
        df_analysis["pred_label"] = le.inverse_transform(df_analysis["y_pred"])

        errors = df_analysis[df_analysis["correct"] == False]

        print("\nEksempler på fejl:")
        print(errors[["true_label", "pred_label", "beta", "snr_db"]].head(20))
