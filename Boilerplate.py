import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression  # udskift/tilføj modeller her

# ============================================================
# 1. LOAD & PREPARE DATA
# ============================================================
df_train = pd.read_csv("Data/TrainingData.csv")

# Fjern irrelevante kolonner
irrelevant_cols = ["filnavn", "beta", "snr_db"]  # tilpas listen
for col in irrelevant_cols:
    if col in df_train.columns:
        df_train = df_train.drop(columns=[col])

# ============================================================
# 2. FILTERING
# ============================================================
# df_train = df_train[...].copy()
# df_train = df_train.dropna(subset=["kolonneNavn"])

# ============================================================
# 3. FEATURES & TARGET
# ============================================================
X = df_train.drop(columns=["target"])
y = df_train["target"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Klasse mapping:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"  {class_name} -> {i}")

# ============================================================
# 4. TRAIN / VALIDATION SPLIT
# ============================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.20, stratify=y_encoded, random_state=42
)

# ============================================================
# 5. EVALUATION FUNCTION (MULTI-CLASS)
# ============================================================
def evaluate_model(model, X_tr, y_tr, X_v, y_v, label):

    # Scaling
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_v_s  = scaler.transform(X_v)

    # Cross-validation (scaled træningsdata)
    cv_scores = cross_val_score(model, X_tr_s, y_tr, cv=5)
    print(f"\n{label} Cross-val accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    # Training
    model.fit(X_tr_s, y_tr)

    # Predictions
    y_pred = model.predict(X_v_s)

    # Print results
    print(f"\n{'='*40}")
    print(f"{label} RESULTS ({model.__class__.__name__})")
    print(f"{'='*40}")
    print(f"Accuracy: {accuracy_score(y_v, y_pred):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_v, y_pred, target_names=label_encoder.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_v, y_pred))

    return model, scaler


# ============================================================
# 6. RUN MODEL (VALIDATION)
# ============================================================
modelNavn = "gay"(class_weight="balanced", max_iter=500)

trained_model, trained_scaler = evaluate_model(
    modelNavn, X_train, y_train, X_val, y_val, "Validation"
)

# ============================================================
# 7. TEST PÅ SEPARAT DATASÆT
# ============================================================
df_test = pd.read_csv("Data/ValidationData.csv")

if "filnavn" in df_test.columns:
    df_test = df_test.drop(columns=["filnavn"])

X_test = df_test.drop(columns=["target", "beta", "snr_db"], errors="ignore")
y_test  = label_encoder.transform(df_test["target"])

X_test_s = trained_scaler.transform(X_test)
y_test_pred = trained_model.predict(X_test_s)

print(f"\n{'='*40}")
print(f"TEST RESULTS ({trained_model.__class__.__name__})")
print(f"{'='*40}")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=label_encoder.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))




