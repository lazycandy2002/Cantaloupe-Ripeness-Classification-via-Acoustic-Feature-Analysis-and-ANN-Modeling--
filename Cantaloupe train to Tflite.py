import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import joblib
import os

# ---------------- Load and Prepare Data ----------------

# Load data
df = pd.read_csv('cantaloupe_features.csv')

# Separate features and labels
X = df.drop(columns=['label']).values
y = df['label'].values  # 1 = ripe, 0 = unripe

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler to use later on Raspberry Pi
joblib.dump(scaler, 'scaler.pkl')

# Split into training and validation
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ---------------- Build the ANN Model ----------------

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ---------------- Train Model ----------------

model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_val, y_val))

# ---------------- Evaluate Model ----------------

loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# ---------------- Save Model ----------------

model.save('cantaloupe_model')  # SavedModel format

# ---------------- Convert to TFLite ----------------

converter = tf.lite.TFLiteConverter.from_saved_model('cantaloupe_model')
tflite_model = converter.convert()

with open('cantaloupe_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ TFLite model saved as 'cantaloupe_model.tflite'")
