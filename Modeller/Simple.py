import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
X_train = df_train.drop(columns=["target"])
y_train = df_train["target"]

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

print("Klasse mapping:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"{class_name} -> {i}")

# Skalering (VIGTIGT for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ----------------------------
# 3. BUILD NEURAL NETWORK
# ----------------------------
input_dim = X_train_scaled.shape[1]
num_classes = len(label_encoder.classes_)

model = Sequential()

# Hidden layer 1
model.add(Dense(
    128,
    input_dim=input_dim,
    activation="relu"
))

# Hidden layer 2
model.add(Dense(
    64,
    activation="relu"
))

# Output layer
model.add(Dense(
    num_classes,
    activation="softmax"
))

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ----------------------------
# 4. TRAIN MODEL (på hele training data)
# ----------------------------
history = model.fit(
    X_train_scaled,
    y_train_encoded,
    epochs=50,
    batch_size=32,
    verbose=1
)

# ----------------------------
# 5. LOAD TEST DATA
# ----------------------------
test_path = "https://raw.githubusercontent.com/marcojs253-crypto/P_2/refs/heads/main/Validering.csv"
df_test = pd.read_csv(test_path)

if "filnavn" in df_test.columns:
    df_test = df_test.drop(columns=["filnavn"])

X_test = df_test.drop(columns=["target"])
y_test = df_test["target"]

# Brug samme encoder og scaler!
y_test_encoded = label_encoder.transform(y_test)
X_test_scaled = scaler.transform(X_test)

print("\nUnikke targets i test:", np.unique(y_test_encoded))
print("Antal prøver pr. klasse:", np.bincount(y_test_encoded))

# ----------------------------
# 6. TEST EVALUATION
# ----------------------------
y_test_pred = np.argmax(model.predict(X_test_scaled), axis=1)

print("\nTest Accuracy:", accuracy_score(y_test_encoded, y_test_pred))
print("\nClassification Report (Test):")
print(classification_report(y_test_encoded, y_test_pred, target_names=label_encoder.classes_))
print("\nConfusion Matrix (Test):")
print(confusion_matrix(y_test_encoded, y_test_pred))

# ----------------------------
# 7. TRAINING CURVE
# ----------------------------
plt.figure(figsize=(10,5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.title("Training Curve (Neural Network)")
plt.legend()
plt.show()