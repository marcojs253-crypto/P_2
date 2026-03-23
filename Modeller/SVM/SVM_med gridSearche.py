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
# ============================================================

# ============================================================
# 0. IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.metrics         import (classification_report,
                                     confusion_matrix,
                                     accuracy_score,
                                     ConfusionMatrixDisplay)
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
df_val   = pd.read_csv(DATA_PATH_VAL)

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
# STEP 7: FEATURE SELECTION  (søg efter optimalt antal features)
# ============================================================
# Features er sorteret efter SHAP importance (faldende).
# Vi prøver at tilføje én feature ad gangen (top-1, top-2, ..., top-N)
# og bruger 5-fold cross-validation KUN på træningsdata til at vælge.
# ValidationData.csv røres ikke her → ingen leakage.
# ============================================================

print("\n" + "="*60)
print("STEP 7: FEATURE SELECTION  (optimalt antal via CV på træningsdata)")
print("="*60)

# Liste over features sorteret efter SHAP importance (faldende)
sorted_features = importance.index.tolist()

# Gem CV accuracy for hvert antal features
acc_per_n = []

print("Søger optimalt antal features via cross-validation på træningsdata...")
for n in range(1, len(sorted_features) + 1):
    # Tag de n vigtigste features
    features_n = sorted_features[:n]

    # Pipeline med scaler + SVM — sikrer ingen leakage i CV-foldene
    pipeline_n = Pipeline([
        ("scaler", StandardScaler()),
        ("svc",    SVC(random_state=RANDOM_STATE, **{
            k.replace("svc__", ""): v
            for k, v in grid_search.best_params_.items()
        })),
    ])

    # 5-fold CV udelukkende på træningsdata
    cv_scores = cross_val_score(
        pipeline_n, X_train[features_n], y_train,
        cv=5, scoring="accuracy", n_jobs=-1
    )
    acc_per_n.append(cv_scores.mean())

# Find det antal features der giver højest accuracy
best_n            = int(np.argmax(acc_per_n)) + 1   # +1 fordi index starter fra 0
selected_features = sorted_features[:best_n]
best_acc_sel      = acc_per_n[best_n - 1]

print(f"Bedste antal features: {best_n} ud af {len(sorted_features)}")
print(f"CV accuracy med {best_n} features: {best_acc_sel:.4f}")
print(f"CV accuracy med alle features:     {grid_search.best_score_:.4f}")
print(f"Valgte features:\n{selected_features}")

# Plot: CV accuracy som funktion af antal features
plt.figure(figsize=(12, 5))
plt.plot(range(1, len(sorted_features) + 1), acc_per_n,
         marker="o", markersize=3, linewidth=1.5, color="steelblue")
plt.axvline(best_n, color="red", linestyle="--", label=f"Optimalt: {best_n} features")
plt.axhline(grid_search.best_score_, color="gray", linestyle=":",
            label=f"CV alle features: {grid_search.best_score_:.4f}")
plt.xlabel("Antal features (sorteret efter SHAP importance)")
plt.ylabel("5-fold CV accuracy  (kun træningsdata)")
plt.title("CV Accuracy vs. antal SHAP-sorterede features")
plt.legend()
plt.tight_layout()
plt.savefig("Modeller/SVM/feature_selection_accuracy.png", dpi=300)
plt.show()
print("Feature selection plot gemt: Modeller/SVM/feature_selection_accuracy.png")

# Visualisér: grøn = valgt, rød = fravalgt
colors = ["green" if f in selected_features else "red" for f in importance.index]
plt.figure(figsize=(14, 6))
plt.bar(importance.index, importance.values, color=colors)
plt.xticks(rotation=90, fontsize=7)
plt.title(f"SHAP Feature Importance – Grøn = Valgt ({best_n} stk.), Rød = Fravalgt", fontsize=13)
plt.xlabel("Feature")
plt.ylabel("Mean |SHAP value|")
plt.tight_layout()
plt.savefig("Modeller/SVM/selected_features.png", dpi=300)
plt.show()
print("Feature selection plot gemt: Modeller/SVM/selected_features.png")

# ============================================================
# STEP 8: NY GRIDSEARCH + GENOPTRÆNING PÅ UDVALGTE FEATURES
# ============================================================
# Hyperparametrene fra step 3 var optimale for alle 43 features.
# Nu kører vi GridSearch igen på de 33 udvalgte features, så
# hyperparametrene er optimale for det reducerede feature-sæt.
# Pipeline sikrer stadig ingen leakage i CV-foldene.
# ============================================================

print("\n" + "="*60)
print("STEP 8: NY GRIDSEARCHCV PÅ UDVALGTE FEATURES")
print("="*60)

# Reducer datasæt til kun de valgte features
X_train_sel = X_train[selected_features]
X_val_sel   = X_val[selected_features]

# Byg ny pipeline til GridSearch på udvalgte features
pipeline_sel = Pipeline([
    ("scaler", StandardScaler()),
    ("svc",    SVC(probability=True, random_state=RANDOM_STATE)),
])

# Kør GridSearch på de udvalgte features
grid_search_sel = GridSearchCV(
    pipeline_sel,
    PARAM_GRID,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
    verbose=1,
)
grid_search_sel.fit(X_train_sel, y_train)

print(f"\nBedste hyperparametre (udvalgte features): {grid_search_sel.best_params_}")
print(f"Bedste CV accuracy    (udvalgte features): {grid_search_sel.best_score_:.4f}")

# Sammenlign de to GridSearch-resultater
print(f"\nHyperparametre step 3 (43 features): {grid_search.best_params_}")
print(f"Hyperparametre step 8 ({len(selected_features)} features): {grid_search_sel.best_params_}")

# Udtræk den bedste pipeline og dens komponenter
best_pipeline_sel = grid_search_sel.best_estimator_
scaler_sel        = best_pipeline_sel.named_steps["scaler"]
svc_sel           = best_pipeline_sel.named_steps["svc"]

# Skalér valideringsdata med scaleren fittet på de udvalgte træningsfeatures
X_val_sel_scaled = scaler_sel.transform(X_val_sel)

print(f"\nModel genoptrænet med {len(selected_features)} features og nye hyperparametre.")

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

# Classification report EFTER
print("\nClassification Report  EFTER feature selection:")
print(classification_report(y_val, y_pred_after, target_names=le.classes_))

# ── Visuel confusion matrix FØR feature selection ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ConfusionMatrixDisplay.from_predictions(
    y_val,
    y_pred_before,
    display_labels=le.classes_,
    cmap="Blues",
    ax=axes[0],
    colorbar=False,
)
axes[0].set_title(f"Confusion Matrix  FØR  feature selection\n"
                  f"(alle {X_train.shape[1]} features,  acc={acc_before:.4f})",
                  fontsize=11)
axes[0].tick_params(axis="x", rotation=45)

# ── Visuel confusion matrix EFTER feature selection ─────────────────────────
ConfusionMatrixDisplay.from_predictions(
    y_val,
    y_pred_after,
    display_labels=le.classes_,
    cmap="Blues",
    ax=axes[1],
    colorbar=False,
)
axes[1].set_title(f"Confusion Matrix  EFTER  feature selection\n"
                  f"({len(selected_features)} features,  acc={acc_after:.4f})",
                  fontsize=11)
axes[1].tick_params(axis="x", rotation=45)

plt.suptitle("SVM – Confusion Matrix før og efter feature selection", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("Modeller/SVM/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()
print("Confusion matrix plot gemt: Modeller/SVM/confusion_matrix.png")
