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
train_path = "/Users/lasseleekrogshave/Desktop/Uni/Semester 2/Gruppeprojekt P2/Training/P2Training.csv"
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

# Feature scaling (VIGTIGT for neural net)
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
num_classes = len(np.unique(y_train))

model = Sequential()

# Hidden layer 1
model.add(Dense(
    neurons := 128,              # Neurons
    input_dim=input_dim,
    activation="relu"            # Propagation function
))

# Hidden layer 2
model.add(Dense(64, activation="relu"))

# Output layer
model.add(Dense(num_classes, activation="softmax"))

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),   # Learning rule
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model
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
print(classification_report(y_val, y_val_pred))
print("\nConfusion Matrix (Validation):")
print(confusion_matrix(y_val, y_val_pred))