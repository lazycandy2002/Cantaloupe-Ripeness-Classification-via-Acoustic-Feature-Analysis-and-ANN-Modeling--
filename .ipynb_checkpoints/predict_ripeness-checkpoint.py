import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the previously saved model
MODEL_PATH = "models/ripeness_classifier.h5"
FEATURES_PATH = "results/acoustic_features.csv"
RESULTS_DIR = "results/"

def extract_features(file_path):
    """
    Extract acoustic features from a single audio file
    Returns a dictionary of features
    """
    try:
        # Load audio file with librosa
        y, sr = librosa.load(file_path, sr=None)
        
        # Basic features
        # MFCC (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_vars = np.var(mfccs, axis=1)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        
        # Compile features in a dictionary
        features = {
            # MFCC means
            **{f'mfcc_mean_{i+1}': mfcc_means[i] for i in range(len(mfcc_means))},
            # MFCC variances
            **{f'mfcc_var_{i+1}': mfcc_vars[i] for i in range(len(mfcc_vars))},
            # Spectral features
            'spectral_centroid_mean': np.mean(spectral_centroid),
            'spectral_centroid_var': np.var(spectral_centroid),
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'spectral_rolloff_var': np.var(spectral_rolloff),
            'spectral_contrast_mean': np.mean(np.mean(spectral_contrast, axis=1)),
            'spectral_contrast_var': np.mean(np.var(spectral_contrast, axis=1)),
            # Zero crossing rate
            'zcr_mean': np.mean(zcr),
            'zcr_var': np.var(zcr),
            # Chroma features
            'chroma_mean': np.mean(np.mean(chroma, axis=1)),
            'chroma_var': np.mean(np.var(chroma, axis=1)),
            # RMS energy
            'rms_mean': np.mean(rms),
            'rms_var': np.var(rms)
        }
        
        return features
        
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        return None

def predict_ripeness(audio_file_path):
    """
    Predict ripeness of a new audio sample
    """
    # Check if file exists
    if not os.path.exists(audio_file_path):
        return f"Error: File {audio_file_path} not found"
        
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        return f"Error: Model file {MODEL_PATH} not found"
    
    # Load the saved model
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Extract features from the audio file
    features = extract_features(audio_file_path)
    
    if features is None:
        return "Error processing audio file"
    
    # Convert to DataFrame for consistent processing
    features_df = pd.DataFrame([features])
    
    # Load the training features to get the correct column order
    if not os.path.exists(FEATURES_PATH):
        return f"Error: Features file {FEATURES_PATH} not found"
        
    training_features = pd.read_csv(FEATURES_PATH)
    
    # Make sure features_df has the same columns as training data (excluding 'label')
    feature_columns = [col for col in training_features.columns if col != 'label']
    
    # Check if all required features are present
    missing_features = [col for col in feature_columns if col not in features_df.columns]
    if missing_features:
        return f"Error: Missing features: {missing_features}"
    
    # Reorder columns to match training data
    features_df = features_df[feature_columns]
    
    # Fit scaler on training data and transform new sample
    scaler = StandardScaler()
    X_train = training_features[feature_columns]
    scaler.fit(X_train)
    features_scaled = scaler.transform(features_df)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0][0]
    
    # Classification result
    result = "Ripe" if prediction > 0.5 else "Unripe"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    return f"Prediction: {result} (confidence: {confidence:.2f})"


    
# At the end of the predict_ripeness.py file, add this code
if __name__ == "__main__":
    import sys
    
    # Check if a single file or directory is provided
    if len(sys.argv) < 2:
        print("Usage: python predict_ripeness.py <path_to_audio_file_or_directory>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # If the path is a directory, process all WAV files in it
    if os.path.isdir(input_path):
        print(f"Processing all WAV files in directory: {input_path}")
        print("-" * 50)
        
        files_processed = 0
        for filename in os.listdir(input_path):
            if filename.endswith('.wav'):
                file_path = os.path.join(input_path, filename)
                result = predict_ripeness(file_path)
                print(f"{filename}: {result}")
                files_processed += 1
        
        if files_processed == 0:
            print(f"No WAV files found in {input_path}")
        else:
            print("-" * 50)
            print(f"Processed {files_processed} files")
    
    # If the path is a single file, just process that file
    elif os.path.isfile(input_path):
        result = predict_ripeness(input_path)
        print(result)
    
    else:
        print(f"Error: {input_path} is not a valid file or directory")