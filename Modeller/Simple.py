import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# ----------------------------
# 1. LOAD TRAINING DATA
# ----------------------------
train_path = "https://raw.githubusercontent.com/marcojs253-crypto/P_2/refs/heads/main/Training.csv"
df_train = pd.read_csv(train_path)

if "filnavn" in df_train.columns:
    df_train = df_train.drop(columns=["filnavn"])

# ----------------------------
# 2. FEATURES & TARGET
# ----------------------------
X = df_train.drop(columns=["target"])
y = df_train["target"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Klasse mapping:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"{class_name} -> {i}")

# Skalering (VIGTIGT for neural networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# 3. TRAIN / VALIDATION SPLIT
# ----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ----------------------------
# 4. NEURAL NETWORK MODEL
# ----------------------------

input_dim = X_train.shape[1]
num_classes = len(label_encoder.classes_)

model = Sequential()

# Hidden layer 1
model.add(Dense(
    128,                 # Neurons
    input_dim=input_dim,
    activation="relu"    # Propagation function
))

# Hidden layer 2
model.add(Dense(
    64,
    activation="relu"
))

# Output layer
model.add(Dense(
    num_classes,
    activation="softmax"  # Multi-class classification
))

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),   # Learning rule
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

# ----------------------------
# 5. VALIDATION EVALUATION
# ----------------------------
y_val_pred = np.argmax(model.predict(X_val), axis=1)

print("\nValidation Accuracy:", accuracy_score(y_val, y_val_pred))
print("\nClassification Report (Validation):")
print(classification_report(y_val, y_val_pred, target_names=label_encoder.classes_))
print("\nConfusion Matrix (Validation):")
print(confusion_matrix(y_val, y_val_pred))

# ----------------------------
# 6. TEST ON SEPARATE DATASET
# ----------------------------
test_path = "https://raw.githubusercontent.com/marcojs253-crypto/P_2/refs/heads/main/Validering.csv"
df_test = pd.read_csv(test_path)

if "filnavn" in df_test.columns:
    df_test = df_test.drop(columns=["filnavn"])

X_test = df_test.drop(columns=["target"])
X_test_scaled = scaler.transform(X_test)

y_test_true = label_encoder.transform(df_test["target"])

print("\nUnikke targets i test:", np.unique(y_test_true))
print("Antal prøver pr. klasse:", np.bincount(y_test_true))

y_test_pred = np.argmax(model.predict(X_test_scaled), axis=1)

print("\nTest Accuracy:", accuracy_score(y_test_true, y_test_pred))
print("\nClassification Report (Test):")
print(classification_report(y_test_true, y_test_pred, target_names=label_encoder.classes_))
print("\nConfusion Matrix (Test):")
print(confusion_matrix(y_test_true, y_test_pred))

# ----------------------------
# 7. VISUALISER LEARNING CURVE
# ----------------------------
plt.figure(figsize=(10,5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.title("Training Curve (Neural Network)")
plt.show()