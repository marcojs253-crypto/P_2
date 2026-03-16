import pandas as pd
import numpy as np
import importlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

# ----------------------------
# 1. LOAD TRAINING DATA
# ----------------------------
train_path = "https://raw.githubusercontent.com/marcojs253-crypto/P_2/refs/heads/main/Data/TrainingData.csv"
df_train = pd.read_csv(train_path)

# Drop irrelevant columns if present
irrelevant_cols = ["filnavn", "beta", "snr_db"]
df_train = df_train.drop(columns=[col for col in irrelevant_cols if col in df_train.columns])

# ----------------------------
# 2. FEATURES TO REMOVE INITIALLY
# ----------------------------
removed_features = [
    "mfcc7_std","mfcc12_std","mfcc9_mean","mfcc10_mean","mfcc10_std",
    "mfcc7_mean","mfcc11_std","mfcc8_std","mfcc13_std","mfcc4_mean",
    "mfcc9_std","mfcc2_std","mfcc3_std","mfcc4_std","mfcc5_std",
    "mfcc6_std","mfcc5_mean","mfcc12_mean","mfcc3_mean","mfcc11_mean",
    "mfcc13_mean","mfcc6_mean","contrast_mean","centroid_mean","flatness_mean",
    "rms_std","flux_std","hnr","centroid_std","rolloff_std","rolloff_mean",
    "contrast_std","flux_mean","mfcc2_mean","mfcc1_mean","zcr_std","bandwidth_std",
    "mfcc8_mean","zcr_mean","mfcc1_std","bandwidth_mean","mfcc1_std"
]

iteration_results = []
iteration = 1

# ----------------------------
# 3. ITERATIVE FEATURE ADDITION
# ----------------------------
while len(removed_features) > 0:
    features_to_add = removed_features[-3:]
    removed_features = removed_features[:-3]

    current_features = [f for f in df_train.columns if f != "target" and f not in removed_features]
    for f in features_to_add:
        if f not in current_features:
            current_features.append(f)

    # Prepare X and y
    X = df_train[current_features]
    y = df_train["target"]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    # ----------------------------
    # 4. DEFINE KERAS MODEL
    # ----------------------------
    def create_model(neurons1=128, neurons2=64, learning_rate=0.001):
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1],)))
        model.add(Dense(neurons1, activation="relu"))
        model.add(Dense(neurons2, activation="relu"))
        model.add(Dense(len(np.unique(y_encoded)), activation="softmax"))
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    clf = KerasClassifier(model=create_model, epochs=50, verbose=0)

    param_grid = {
        "model__neurons1": [64, 128],
        "model__neurons2": [32, 64],
        "model__learning_rate": [0.001, 0.005],
        "batch_size": [16, 32]
    }

    # Grid search
    grid = GridSearchCV(clf, param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # ----------------------------
    # 5. VALIDATION METRICS
    # ----------------------------
    y_val_pred = np.argmax(best_model.model_.predict(X_val), axis=1)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_prec = precision_score(y_val, y_val_pred, average="macro")
    val_rec = recall_score(y_val, y_val_pred, average="macro")
    val_f1 = f1_score(y_val, y_val_pred, average="macro")

    # ----------------------------
    # 6. TEST METRICS
    # ----------------------------
    test_path = "https://raw.githubusercontent.com/marcojs253-crypto/P_2/refs/heads/main/Data/ValidationData.csv"
    df_test = pd.read_csv(test_path)
    if "filnavn" in df_test.columns:
        df_test = df_test.drop(columns=["filnavn"])
    
    X_test = df_test.drop(columns=["target", "beta", "snr_db"] + removed_features, errors="ignore")
    for f in features_to_add:
        if f in df_test.columns and f not in X_test.columns:
            X_test[f] = df_test[f]

    X_test_scaled = scaler.transform(X_test)
    y_test_true = label_encoder.transform(df_test["target"])
    y_test_pred = np.argmax(best_model.model_.predict(X_test_scaled), axis=1)
    test_acc = accuracy_score(y_test_true, y_test_pred)
    test_prec = precision_score(y_test_true, y_test_pred, average="macro")
    test_rec = recall_score(y_test_true, y_test_pred, average="macro")
    test_f1 = f1_score(y_test_true, y_test_pred, average="macro")

    # ----------------------------
    # 7. SAVE ITERATION RESULTS
    # ----------------------------
    iteration_results.append({
        "Iteration": iteration,
        "Added_Features": features_to_add,
        "Num_Features": len(current_features),
        "Val_Accuracy": val_acc,
        "Val_Precision": val_prec,
        "Val_Recall": val_rec,
        "Val_F1": val_f1,
        "Test_Accuracy": test_acc,
        "Test_Precision": test_prec,
        "Test_Recall": test_rec,
        "Test_F1": test_f1,
        "Best_Params": grid.best_params_
    })

    print(f"Iteration {iteration}: added {features_to_add}")
    print(f"  Validation Accuracy: {val_acc:.4f}, Test Accuracy: {test_acc:.4f}")

    iteration += 1

# ----------------------------
# 8. SUMMARY DATAFRAME
# ----------------------------
summary_df = pd.DataFrame(iteration_results)
print("\nSummary of all iterations:")
print(summary_df)

# ----------------------------
# 9. VISUALIZE METRICS
# ----------------------------
metrics_val = ["Val_Accuracy", "Val_Precision", "Val_Recall", "Val_F1"]
metrics_test = ["Test_Accuracy", "Test_Precision", "Test_Recall", "Test_F1"]

plt.figure(figsize=(14, 6))
for metric in metrics_val:
    plt.plot(summary_df["Iteration"], summary_df[metric], marker='o', label=metric)
for metric in metrics_test:
    plt.plot(summary_df["Iteration"], summary_df[metric], marker='x', label=metric)
plt.xticks(summary_df["Iteration"])
plt.xlabel("Iteration (Added Features)")
plt.ylabel("Score")
plt.title("Validation (o) and Test (x) Metrics per Iteration")
plt.grid(True)
plt.legend()
plt.show()