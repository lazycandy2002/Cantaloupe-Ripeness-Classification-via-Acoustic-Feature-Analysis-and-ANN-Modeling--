import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# --- Paths & Params ---
RIPE_DIR    = "data/ripe"
UNRIPE_DIR  = "data/unripe"
RESULTS_DIR = "results"
MODEL_PATH  = "models/ripeness_classifier.h5"
SCALER_PATH = "models/scaler.pkl"
FEATURE_CSV = os.path.join(RESULTS_DIR, "acoustic_features.csv")
THRESHOLD   = 0.6

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# --- 1. Augmentation ---
def augment_audio(y, sr):
    out = []
    out.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=2))  # Corrected the call
    out.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)) # Corrected the call
    out.append(librosa.effects.time_stretch(y, rate=0.9))
    out.append(librosa.effects.time_stretch(y, rate=1.1))
    noise = 0.005 * np.random.randn(len(y))
    out.append(y + noise)
    return out

# --- 2. Feature Extraction ---
def extract_features(y, sr):
    mfccs     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    contrast  = librosa.feature.spectral_contrast(y=y, sr=sr)
    chroma    = librosa.feature.chroma_stft(y=y, sr=sr)
    zcr       = librosa.feature.zero_crossing_rate(y)[0]
    rms       = librosa.feature.rms(y=y)[0]

    feat = {}
    for i in range(13):
        feat[f"mfcc_mean_{i+1}"] = np.mean(mfccs[i])
        feat[f"mfcc_var_{i+1}"]  = np.var(mfccs[i])
    feat.update({
        "spectral_centroid_mean": np.mean(centroid),
        "spectral_centroid_var":  np.var(centroid),
        "spectral_rolloff_mean":  np.mean(rolloff),
        "spectral_rolloff_var":   np.var(rolloff),
        "spectral_contrast_mean": np.mean(contrast),
        "spectral_contrast_var":  np.var(contrast),
        "chroma_mean":            np.mean(chroma),
        "chroma_var":             np.var(chroma),
        "zcr_mean":               np.mean(zcr),
        "zcr_var":                np.var(zcr),
        "rms_mean":               np.mean(rms),
        "rms_var":                np.var(rms),
    })
    return feat

# --- 3. Build Dataset ---
def build_dataset():
    feats, labels = [], []
    for label, directory in [(1, RIPE_DIR), (0, UNRIPE_DIR)]:
        for fname in os.listdir(directory):
            if not fname.lower().endswith(".wav"):
                continue
            path = os.path.join(directory, fname)
            y, sr = librosa.load(path, sr=None)
            feats.append(extract_features(y, sr)); labels.append(label)
            for aug in augment_audio(y, sr):
                feats.append(extract_features(aug, sr)); labels.append(label)
    df = pd.DataFrame(feats)
    df["label"] = labels
    df.to_csv(FEATURE_CSV, index=False)
    return df

# --- 4. Prepare Data ---
def prepare_data(df):
    X = df.drop("label", axis=1); y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler().fit(X_train)
    joblib.dump(scaler, SCALER_PATH)
    return scaler, scaler.transform(X_train), scaler.transform(X_test), y_train, y_test

# --- 5. Model ---
def create_model(input_dim):
    m = Sequential([
        Dense(128, activation="relu", input_dim=input_dim), Dropout(0.4),
        Dense(64,  activation="relu"),                    Dropout(0.3),
        Dense(32,  activation="relu"),                    Dropout(0.2),
        Dense(1,   activation="sigmoid")
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m

# --- 6. Train ---
def train_model(model, X_train, y_train):
    w = class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cw = {i: w_i for i, w_i in enumerate(w)}
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, batch_size=16,
              validation_split=0.2, class_weight=cw, callbacks=[es], verbose=1)
    model.save(MODEL_PATH)
    return model

# --- 7. Evaluate ---
def evaluate(model, X_test, y_test):
    probs = model.predict(X_test).flatten()
    fpr, tpr, thr = roc_curve(y_test, probs)
    best = thr[np.argmax(tpr - fpr)]
    print(f"Default th: {THRESHOLD}, ROC best th: {best:.2f}")
    for th in (THRESHOLD, best):
        preds = (probs > th).astype(int)
        print(f"\n=== Metrics @ th={th:.2f} ===")
        print(classification_report(y_test, preds, target_names=["Unripe","Ripe"]))
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Unripe","Ripe"], yticklabels=["Unripe","Ripe"])
        plt.title(f"Confusion Matrix (th={th:.2f})"); plt.show()

if __name__ == "__main__":
    df = build_dataset()
    scaler, X_train, X_test, y_train, y_test = prepare_data(df)
    model = create_model(X_train.shape[1])
    model = train_model(model, X_train, y_train)
    evaluate(model, X_test, y_test)
