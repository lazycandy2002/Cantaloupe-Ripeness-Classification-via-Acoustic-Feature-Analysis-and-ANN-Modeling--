import os
import numpy as np
import pandas as pd
import librosa
import joblib
import tensorflow as tf

# --- Paths & Params ---
MODEL_PATH  = "models/ripeness_classifier.h5"
SCALER_PATH = "models/scaler.pkl"
SAMPLES_DIR = "samples"
THRESHOLD   = 0.6

# --- Feature Extraction (same as training) ---
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

# --- Load model & scaler ---
model  = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# --- Predict function ---
def predict_ripeness(path):
    y, sr = librosa.load(path, sr=None)
    feats = extract_features(y, sr)
    X = scaler.transform(pd.DataFrame([feats]))
    prob = model.predict(X)[0][0]
    label = "Ripe" if prob > THRESHOLD else "Unripe"
    return label, prob

# --- Run on all samples ---
if __name__ == "__main__":
    print(f"🧪 Testing samples in '{SAMPLES_DIR}':\n")
    for fname in sorted(os.listdir(SAMPLES_DIR)):
        if fname.lower().endswith(".wav"):
            path = os.path.join(SAMPLES_DIR, fname)
            label, prob = predict_ripeness(path)
            print(f"{fname} → {label} (conf: {prob:.2f})")
