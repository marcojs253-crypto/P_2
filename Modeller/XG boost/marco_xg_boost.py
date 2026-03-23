# ============================================================
# XGBOOST KLASSIFIKATIONSMODEL - STØJTYPE GENKENDELSE
# ============================================================
# Rækkefølge:
#   1.  Load og rens data  (TrainingData.csv + ValidationData.csv)
#   2.  LabelEncoder på target
#   3.  GridSearchCV  (5-fold CV)
#   4.  Print bedste hyperparametre + CV accuracy
#   5.  Evaluér på ValidationData.csv  (alle features)
#   6.  Feature importance  (XGBoost indbygget)
#   7.  Feature selection  (85 % kumulativ importance)
#   8.  Ny GridSearchCV på udvalgte features  (re-optimerede hyperparametre)
#   9.  Sammenlign accuracy FØR / EFTER feature selection
# ============================================================

# ============================================================
# 0. IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import (classification_report,
                                     confusion_matrix,
                                     accuracy_score,
                                     ConfusionMatrixDisplay)

from xgboost import XGBClassifier

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

# Hyperparameter-grid til GridSearchCV.
# XGBoost er træbaseret og kræver ikke StandardScaler.
PARAM_GRID = {
    "n_estimators":     [100, 200, 300],
    "max_depth":        [3, 5, 7],
    "learning_rate":    [0.05, 0.1, 0.2],
    "gamma":            [0, 0.1, 0.3],
    "subsample":        [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

# ============================================================
# STEP 1: LOAD DATA
# ============================================================

print("\n" + "="*60)
print("STEP 1: LOAD DATA")
print("="*60)

df_train = pd.read_csv(DATA_PATH_TRAIN)
df_val   = pd.read_csv(DATA_PATH_VAL)

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
    print(f"ADVARSEL: Imbalance ratio = {imbalance_ratio:.2f} – overvej scale_pos_weight")
else:
    print(f"Datasættet er balanceret  (ratio = {imbalance_ratio:.2f})")

# ============================================================
# STEP 3: GRIDSEARCHCV  (5-fold CV)
# ============================================================
# XGBoost er træbaseret → ingen StandardScaler nødvendig.
# GridSearchCV med 5-fold CV direkte på XGBClassifier.
# ============================================================

print("\n" + "="*60)
print("STEP 3: GRIDSEARCHCV  (5-fold CV)")
print("="*60)

base_model = XGBClassifier(
    objective    = "multi:softprob",
    num_class    = len(le.classes_),
    eval_metric  = "mlogloss",
    tree_method  = "hist",
    random_state = RANDOM_STATE,
)

grid_search = GridSearchCV(
    base_model,
    PARAM_GRID,
    scoring  = "accuracy",
    cv       = 5,
    n_jobs   = -1,
    verbose  = 1,
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

best_model = grid_search.best_estimator_

# ============================================================
# STEP 5: EVALUERING PÅ VALIDERINGSDATA  (alle features)
# ============================================================

print("\n" + "="*60)
print("STEP 5: EVALUERING PÅ VALIDERINGSDATA  (alle features)")
print("="*60)

y_pred_before = best_model.predict(X_val)
acc_before    = accuracy_score(y_val, y_pred_before)

print(f"Accuracy (alle features): {acc_before:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred_before, target_names=le.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred_before))

# ============================================================
# STEP 6: FEATURE IMPORTANCE  (XGBoost indbygget)
# ============================================================
# XGBoost beregner feature importance baseret på Gini gain –
# ingen ekstern forklarer nødvendig.
# ============================================================

print("\n" + "="*60)
print("STEP 6: FEATURE IMPORTANCE  (XGBoost indbygget)")
print("="*60)

feature_names = list(X_train.columns)
importance    = pd.Series(
    best_model.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

print("\nTop 10 vigtigste features:")
print(importance.head(10).to_string())

# Plot: søjlediagram over feature importance
plt.figure(figsize=(14, 6))
plt.bar(importance.index, importance.values, color="steelblue")
plt.xticks(rotation=90, fontsize=7)
plt.title("Feature Importance (XGBoost) – Gini Gain per feature", fontsize=13)
plt.xlabel("Feature")
plt.ylabel("Feature Importance")
plt.tight_layout()
plt.savefig("Modeller/XG boost/feature_importance.png", dpi=300)
plt.show()
print("Feature importance plot gemt: Modeller/XG boost/feature_importance.png")

# ============================================================
# STEP 7: FEATURE SELECTION  (85 % kumulativ importance)
# ============================================================
# Features er sorteret efter importance (faldende).
# Vi vælger de features der tilsammen udgør 85 % af den samlede
# importance – den mindste mængde features der forklarer
# størsteparten af modellens beslutningsgrundlag.
# ValidationData.csv røres ikke her → ingen leakage.
# ============================================================

print("\n" + "="*60)
print(f"STEP 7: FEATURE SELECTION  ({int(IMPORTANCE_THRESHOLD*100)} % kumulativ importance)")
print("="*60)

sorted_features = importance.index.tolist()

# Beregn kumulativ importance som andel af total
cumulative = importance.cumsum() / importance.sum()

# Vælg features indtil tærsklen nås
selected_features = cumulative[cumulative <= IMPORTANCE_THRESHOLD].index.tolist()

# Tilføj den feature der krydser tærsklen
next_idx = len(selected_features)
if next_idx < len(importance):
    selected_features.append(importance.index[next_idx])

best_n = len(selected_features)

print(f"Valgte {best_n} ud af {len(sorted_features)} features  "
      f"(dækker {cumulative.iloc[best_n - 1]:.1%} af importance)")
print(f"Valgte features:\n{selected_features}")

# Plot: kumulativ importance med tærskel
plt.figure(figsize=(12, 5))
plt.plot(range(1, len(sorted_features) + 1), cumulative.values,
         marker="o", markersize=3, linewidth=1.5, color="steelblue")
plt.axvline(best_n, color="red", linestyle="--", label=f"Valgt: {best_n} features")
plt.axhline(IMPORTANCE_THRESHOLD, color="gray", linestyle=":",
            label=f"Tærskel: {int(IMPORTANCE_THRESHOLD*100)} %")
plt.xlabel("Antal features (sorteret efter importance)")
plt.ylabel("Kumulativ importance (andel)")
plt.title(f"Kumulativ Feature Importance – {int(IMPORTANCE_THRESHOLD*100)} % tærskel")
plt.legend()
plt.tight_layout()
plt.savefig("Modeller/XG boost/feature_selection_accuracy.png", dpi=300)
plt.show()
print("Feature selection plot gemt: Modeller/XG boost/feature_selection_accuracy.png")

# Visualisér: grøn = valgt, rød = fravalgt
colors = ["green" if f in selected_features else "red" for f in importance.index]
plt.figure(figsize=(14, 6))
plt.bar(importance.index, importance.values, color=colors)
plt.xticks(rotation=90, fontsize=7)
plt.title(f"Feature Importance – Grøn = Valgt ({best_n} stk.), Rød = Fravalgt", fontsize=13)
plt.xlabel("Feature")
plt.ylabel("Feature Importance")
plt.tight_layout()
plt.savefig("Modeller/XG boost/selected_features.png", dpi=300)
plt.show()
print("Feature selection plot gemt: Modeller/XG boost/selected_features.png")

# ============================================================
# STEP 8: NY GRIDSEARCHCV PÅ UDVALGTE FEATURES
# ============================================================
# Hyperparametrene fra step 3 var optimale for alle features.
# Nu kører vi GridSearch igen på de udvalgte features, så
# hyperparametrene er optimale for det reducerede feature-sæt.
# ============================================================

print("\n" + "="*60)
print("STEP 8: NY GRIDSEARCHCV PÅ UDVALGTE FEATURES")
print("="*60)

X_train_sel = X_train[selected_features]
X_val_sel   = X_val[selected_features]

base_model_sel = XGBClassifier(
    objective    = "multi:softprob",
    num_class    = len(le.classes_),
    eval_metric  = "mlogloss",
    tree_method  = "hist",
    random_state = RANDOM_STATE,
)

grid_search_sel = GridSearchCV(
    base_model_sel,
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

best_model_sel = grid_search_sel.best_estimator_
print(f"\nModel genoptrænet med {best_n} features og nye hyperparametre.")

# ============================================================
# STEP 9: SAMMENLIGNING FØR OG EFTER FEATURE SELECTION
# ============================================================

print("\n" + "="*60)
print("STEP 9: SAMMENLIGNING FØR / EFTER FEATURE SELECTION")
print("="*60)

y_pred_after = best_model_sel.predict(X_val_sel)
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

plt.suptitle("XGBoost – Confusion Matrix før og efter feature selection", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("Modeller/XG boost/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()
print("Confusion matrix plot gemt: Modeller/XG boost/confusion_matrix.png")
