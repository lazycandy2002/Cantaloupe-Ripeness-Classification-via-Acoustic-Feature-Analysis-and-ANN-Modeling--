import os
import tensorflow as tf
import argparse

def convert_model_to_tflite(input_path, output_path):
    """Convert TensorFlow model to TensorFlow Lite model"""
    
    print(f"Loading model from {input_path}...")
    model = tf.keras.models.load_model(input_path)
    
    print("Converting model to TFLite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    print(f"Saving TFLite model to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print("Conversion complete!")
    print(f"Original model size: {os.path.getsize(input_path) / (1024 * 1024):.2f} MB")
    print(f"TFLite model size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TensorFlow model to TFLite")
    parser.add_argument("--input", default="models/ripeness_classifier.h5", 
                        help="Path to input TensorFlow model")
    parser.add_argument("--output", default="models/ripeness_classifier.tflite", 
                        help="Path to output TFLite model")
    
    args = parser.parse_args()
    convert_model_to_tflite(args.input, args.output)