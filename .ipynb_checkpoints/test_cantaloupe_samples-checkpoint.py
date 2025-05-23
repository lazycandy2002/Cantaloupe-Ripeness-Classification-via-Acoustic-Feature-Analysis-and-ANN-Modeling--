import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# --- Paths ---
MODEL_PATH = "models/ripeness_classifier.h5"
FEATURE_CSV = "results/acoustic_features.csv"
SAMPLES_DIR = "samples"

# --- Feature Extraction Function ---
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

        features = {
            'mfcc_mean': np.mean(mfccs),
            'mfcc_std': np.std(mfccs),
            'chroma_mean': np.mean(chroma),
            'chroma_std': np.std(chroma),
            'zcr_mean': np.mean(zcr),
            'zcr_std': np.std(zcr),
            'centroid_mean': np.mean(spectral_centroid),
            'centroid_std': np.std(spectral_centroid),
        }

        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# --- Predict Ripeness ---
def predict_ripeness(audio_file_path, model, scaler, feature_csv):
    features = extract_features(audio_file_path)
    if features is None:
        return "Error"

    features_df = pd.DataFrame([features])
    features_scaled = scaler.transform(features_df)
    prediction = model.predict(features_scaled)[0][0]

    label = "Ripe" if prediction > 0.5 else "Unripe"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return f"{label} (confidence: {confidence:.2f})"

# --- Load Model & Scaler ---
model = tf.keras.models.load_model(MODEL_PATH)
feature_data = pd.read_csv(FEATURE_CSV)
scaler = StandardScaler()
scaler.fit(feature_data.drop(columns=["label"]))

# --- Test All Samples ---
print(f"\n🧪 Testing samples in '{SAMPLES_DIR}'...\n")
for fname in sorted(os.listdir(SAMPLES_DIR)):
    if fname.endswith(".wav"):
        fpath = os.path.join(SAMPLES_DIR, fname)
        result = predict_ripeness(fpath, model, scaler, FEATURE_CSV)
        print(f"{fname} → {result}")
