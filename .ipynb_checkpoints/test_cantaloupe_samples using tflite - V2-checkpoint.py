import os
import numpy as np
import pandas as pd
import librosa
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from sklearn.metrics import classification_report
import re
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# --- Paths & Params ---
MODEL_PATH = "models/ripeness_classifier.tflite"
SCALER_PATH = "models/scaler.pkl"
SAMPLES_DIR = "samples"
THRESHOLD = 0.6

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# --- Feature Extraction (same as training) ---
def extract_features(y, sr):
    """Extract audio features for classification"""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]

    feat = {}
    for i in range(13):
        feat[f"mfcc_mean_{i+1}"] = np.mean(mfccs[i])
        feat[f"mfcc_var_{i+1}"] = np.var(mfccs[i])
    feat.update({
        "spectral_centroid_mean": np.mean(centroid),
        "spectral_centroid_var": np.var(centroid),
        "spectral_rolloff_mean": np.mean(rolloff),
        "spectral_rolloff_var": np.var(rolloff),
        "spectral_contrast_mean": np.mean(contrast),
        "spectral_contrast_var": np.var(contrast),
        "chroma_mean": np.mean(chroma),
        "chroma_var": np.var(chroma),
        "zcr_mean": np.mean(zcr),
        "zcr_var": np.var(zcr),
        "rms_mean": np.mean(rms),
        "rms_var": np.var(rms),
    })
    return feat

# --- Load TFLite model & scaler ---
print("Loading model and scaler...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
scaler = joblib.load(SCALER_PATH)

# --- Predict function ---
def predict_ripeness(path):
    """Predict ripeness from audio file"""
    y, sr = librosa.load(path, sr=None)
    feats = extract_features(y, sr)
    X = scaler.transform(pd.DataFrame([feats]))
    
    input_data = X.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prob = output_data[0][0]
    
    label = "Ripe" if prob > THRESHOLD else "Unripe"
    return label, prob

# --- Parse filename to get true label ---
def parse_filename(filename):
    """Extract true label from filename (expects format like 'ripe1.wav' or 'unripe2.wav')"""
    filename_lower = filename.lower()
    if filename_lower.startswith('ripe'):
        return 'Ripe'
    elif filename_lower.startswith('unripe'):
        return 'Unripe'
    else:
        # Try to find ripe/unripe anywhere in filename
        if 'ripe' in filename_lower and 'unripe' not in filename_lower:
            return 'Ripe'
        elif 'unripe' in filename_lower:
            return 'Unripe'
        else:
            return 'Unknown'

# --- Analysis Class ---
class RipenessAnalyzer:
    def __init__(self):
        self.results = []
        self.predictions = []
        self.true_labels = []
        self.probabilities = []
        self.filenames = []
        
    def analyze_samples(self, samples_dir):
        """Analyze all samples in directory"""
        print(f"🧪 Analyzing samples in '{samples_dir}':\n")
        
        audio_files = [f for f in sorted(os.listdir(samples_dir)) 
                      if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a'))]
        
        if not audio_files:
            print("No audio files found!")
            return
            
        for fname in audio_files:
            path = os.path.join(samples_dir, fname)
            try:
                predicted_label, prob = predict_ripeness(path)
                true_label = parse_filename(fname)
                
                self.filenames.append(fname)
                self.predictions.append(predicted_label)
                self.true_labels.append(true_label)
                self.probabilities.append(prob)
                
                # Determine if prediction is correct
                correct = "✅" if predicted_label == true_label else "❌"
                if true_label == "Unknown":
                    correct = "❓"
                
                print(f"{fname} → {predicted_label} (conf: {prob:.3f}) | True: {true_label} {correct}")
                
                self.results.append({
                    'filename': fname,
                    'predicted': predicted_label,
                    'true_label': true_label,
                    'probability': prob,
                    'correct': predicted_label == true_label if true_label != "Unknown" else None
                })
                
            except Exception as e:
                print(f"Error processing {fname}: {e}")
    
    def calculate_metrics(self):
        """Calculate classification metrics"""
        if not self.results:
            print("No results to analyze!")
            return None
            
        # Filter out unknown labels for metrics calculation
        valid_results = [r for r in self.results if r['true_label'] != 'Unknown']
        
        if not valid_results:
            print("No labeled samples found for metrics calculation!")
            return None
            
        y_true = [r['true_label'] for r in valid_results]
        y_pred = [r['predicted'] for r in valid_results]
        y_prob = [r['probability'] for r in valid_results]
        
        # Convert to binary for metrics
        y_true_bin = [1 if label == 'Ripe' else 0 for label in y_true]
        y_pred_bin = [1 if label == 'Ripe' else 0 for label in y_pred]
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        print(f"\n📊 CLASSIFICATION METRICS:")
        print(f"{'='*50}")
        print(f"Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1-Score:  {f1:.3f}")
        print(f"Samples:   {len(valid_results)}")
        
        # Detailed classification report
        print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
        print(f"{'='*50}")
        print(classification_report(y_true, y_pred))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_true': y_true_bin,
            'y_pred': y_pred_bin,
            'y_prob': y_prob,
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
    
    def plot_results(self, metrics=None):
        """Create individual visualization plots and save as separate PNG files"""
        if not self.results:
            print("No results to plot!")
            return
            
        # Create output directory for graphs
        os.makedirs('analysis_graphs', exist_ok=True)
        saved_files = []
        
        # 1. Probability Distribution
        plt.figure(figsize=(10, 6))
        ripe_probs = [r['probability'] for r in self.results if r['predicted'] == 'Ripe']
        unripe_probs = [r['probability'] for r in self.results if r['predicted'] == 'Unripe']
        
        plt.hist(ripe_probs, alpha=0.7, label=f'Predicted Ripe (n={len(ripe_probs)})', 
                bins=20, color='green', edgecolor='black')
        plt.hist(unripe_probs, alpha=0.7, label=f'Predicted Unripe (n={len(unripe_probs)})', 
                bins=20, color='red', edgecolor='black')
        plt.axvline(x=THRESHOLD, color='black', linestyle='--', linewidth=2, 
                   label=f'Threshold ({THRESHOLD})')
        plt.xlabel('Prediction Probability', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Probability Distribution of Predictions', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        filename = 'analysis_graphs/01_probability_distribution.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        saved_files.append(filename)
        
        # 2. Confidence vs Accuracy
        plt.figure(figsize=(10, 6))
        valid_results = [r for r in self.results if r['true_label'] != 'Unknown']
        if valid_results:
            probs = [r['probability'] for r in valid_results]
            correct_vals = [1 if r['correct'] else 0 for r in valid_results]
            colors = ['green' if r['correct'] else 'red' for r in valid_results]
            
            plt.scatter(probs, correct_vals, c=colors, alpha=0.7, s=60, edgecolors='black')
            plt.xlabel('Prediction Probability', fontsize=12)
            plt.ylabel('Correct Prediction (1=Correct, 0=Wrong)', fontsize=12)
            plt.title('Confidence vs Accuracy Scatter Plot', fontsize=14, fontweight='bold')
            plt.ylim(-0.1, 1.1)
            plt.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(probs, correct_vals, 1)
            p = np.poly1d(z)
            plt.plot(probs, p(probs), "b--", alpha=0.8, linewidth=2, label=f'Trend line')
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No labeled data available', ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=14)
        plt.tight_layout()
        filename = 'analysis_graphs/02_confidence_vs_accuracy.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        saved_files.append(filename)
        
        # 3. Confusion Matrix
        if metrics and 'confusion_matrix' in metrics:
            plt.figure(figsize=(8, 6))
            cm = metrics['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'},
                       xticklabels=['Unripe', 'Ripe'], 
                       yticklabels=['Unripe', 'Ripe'],
                       square=True, linewidths=1)
            plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            
            # Add accuracy annotation
            accuracy = np.trace(cm) / np.sum(cm)
            plt.figtext(0.02, 0.02, f'Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)', 
                       ha='left', va='bottom', fontsize=11, 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            plt.tight_layout()
            filename = 'analysis_graphs/03_confusion_matrix.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            saved_files.append(filename)
        
        # 4. ROC Curve
        if metrics and len(set(metrics['y_true'])) > 1:
            plt.figure(figsize=(8, 8))
            fpr, tpr, thresholds = roc_curve(metrics['y_true'], metrics['y_prob'])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color='darkorange', lw=3, 
                    label=f'ROC Curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                    label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curve - Receiver Operating Characteristic', fontsize=14, fontweight='bold')
            plt.legend(loc="lower right", fontsize=11)
            plt.grid(True, alpha=0.3)
            
            # Find optimal threshold point
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100, 
                       label=f'Optimal Point (threshold={optimal_threshold:.3f})', zorder=5)
            plt.legend(loc="lower right", fontsize=11)
            plt.tight_layout()
            filename = 'analysis_graphs/04_roc_curve.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            saved_files.append(filename)
        
        # 5. Prediction Counts
        plt.figure(figsize=(8, 6))
        pred_counts = pd.Series([r['predicted'] for r in self.results]).value_counts()
        colors = ['green' if idx == 'Ripe' else 'red' for idx in pred_counts.index]
        bars = plt.bar(pred_counts.index, pred_counts.values, color=colors, alpha=0.8, 
                       edgecolor='black', linewidth=1.5)
        plt.title('Prediction Distribution', fontsize=14, fontweight='bold')
        plt.ylabel('Count', fontsize=12)
        plt.xlabel('Predicted Class', fontsize=12)
        
        # Add value labels on bars
        for bar, count in zip(bars, pred_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Add percentage labels
        total = sum(pred_counts.values)
        for bar, count in zip(bars, pred_counts.values):
            percentage = (count/total)*100
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                    f'{percentage:.1f}%', ha='center', va='center', 
                    fontsize=11, color='white', fontweight='bold')
        plt.tight_layout()
        filename = 'analysis_graphs/05_prediction_counts.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        saved_files.append(filename)
        
        # 6. True Label Distribution
        plt.figure(figsize=(8, 6))
        true_counts = pd.Series([r['true_label'] for r in self.results 
                               if r['true_label'] != 'Unknown']).value_counts()
        if not true_counts.empty:
            colors = ['green' if idx == 'Ripe' else 'red' for idx in true_counts.index]
            bars = plt.bar(true_counts.index, true_counts.values, color=colors, alpha=0.8,
                          edgecolor='black', linewidth=1.5)
            plt.title('True Label Distribution', fontsize=14, fontweight='bold')
            plt.ylabel('Count', fontsize=12)
            plt.xlabel('True Class', fontsize=12)
            
            # Add value labels
            for bar, count in zip(bars, true_counts.values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            # Add percentage labels
            total = sum(true_counts.values)
            for bar, count in zip(bars, true_counts.values):
                percentage = (count/total)*100
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                        f'{percentage:.1f}%', ha='center', va='center', 
                        fontsize=11, color='white', fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'No labeled data available\n(rename files to ripe*.wav or unripe*.wav)', 
                    ha='center', va='center', transform=plt.gca().transAxes, 
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            plt.title('True Label Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        filename = 'analysis_graphs/06_true_label_distribution.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        saved_files.append(filename)
        
        # 7. Accuracy by Confidence Bins
        plt.figure(figsize=(10, 6))
        valid_results = [r for r in self.results if r['true_label'] != 'Unknown']
        if valid_results:
            # Create confidence bins
            bins = np.linspace(0, 1, 6)  # 5 bins
            bin_accuracies = []
            bin_centers = []
            bin_counts = []
            bin_labels = []
            
            for i in range(len(bins)-1):
                bin_results = [r for r in valid_results 
                             if bins[i] <= r['probability'] < bins[i+1]]
                if bin_results:
                    accuracy = sum(r['correct'] for r in bin_results) / len(bin_results)
                    bin_accuracies.append(accuracy)
                    bin_centers.append((bins[i] + bins[i+1]) / 2)
                    bin_counts.append(len(bin_results))
                    bin_labels.append(f'{bins[i]:.1f}-{bins[i+1]:.1f}')
            
            if bin_accuracies:
                bars = plt.bar(range(len(bin_accuracies)), bin_accuracies, 
                              color='skyblue', alpha=0.8, edgecolor='black', linewidth=1.5)
                plt.xlabel('Confidence Bins', fontsize=12)
                plt.ylabel('Accuracy', fontsize=12)
                plt.title('Accuracy by Confidence Level', fontsize=14, fontweight='bold')
                plt.ylim(0, 1.1)
                plt.xticks(range(len(bin_labels)), bin_labels)
                
                # Add accuracy and count labels
                for i, (bar, accuracy, count) in enumerate(zip(bars, bin_accuracies, bin_counts)):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                            f'{accuracy:.2f}', ha='center', va='bottom', 
                            fontsize=11, fontweight='bold')
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                            f'n={count}', ha='center', va='center', 
                            fontsize=10, color='darkblue')
                plt.grid(True, alpha=0.3, axis='y')
        else:
            plt.text(0.5, 0.5, 'No labeled data available for accuracy analysis', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.tight_layout()
        filename = 'analysis_graphs/07_accuracy_by_confidence.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        saved_files.append(filename)
        
        # 8. Accuracy Summary Chart
        if metrics:
            plt.figure(figsize=(10, 6))
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            metric_values = [metrics['accuracy'], metrics['precision'], 
                           metrics['recall'], metrics['f1']]
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            bars = plt.bar(metric_names, metric_values, color=colors, alpha=0.8, 
                          edgecolor='black', linewidth=1.5)
            
            plt.ylim(0, 1.1)
            plt.ylabel('Score', fontsize=12)
            plt.title('Model Performance Metrics', fontsize=14, fontweight='bold')
            
            # Add value labels
            for bar, value in zip(bars, metric_values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                        f'{value:.3f}', ha='center', va='bottom', 
                        fontsize=12, fontweight='bold')
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                        f'{value*100:.1f}%', ha='center', va='center', 
                        fontsize=11, color='white', fontweight='bold')
            
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            filename = 'analysis_graphs/08_accuracy_summary.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            saved_files.append(filename)
        
        print(f"\n📈 Individual graphs saved to 'analysis_graphs/' folder:")
        for i, file in enumerate(saved_files, 1):
            print(f"   {i}. {os.path.basename(file)}")
        
        return saved_files
    
    def save_detailed_results(self):
        """Save detailed results to CSV"""
        if not self.results:
            print("No results to save!")
            return
            
        df = pd.DataFrame(self.results)
        df.to_csv('ripeness_results.csv', index=False)
        print(f"📄 Detailed results saved to 'ripeness_results.csv'")
        
        # Summary statistics
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"{'='*50}")
        print(f"Total samples analyzed: {len(self.results)}")
        
        labeled_results = [r for r in self.results if r['true_label'] != 'Unknown']
        if labeled_results:
            correct = sum(1 for r in labeled_results if r['correct'])
            print(f"Labeled samples: {len(labeled_results)}")
            print(f"Correct predictions: {correct}")
            print(f"Overall accuracy: {correct/len(labeled_results):.3f} ({correct/len(labeled_results)*100:.1f}%)")
        
        print(f"Average confidence: {np.mean([r['probability'] for r in self.results]):.3f}")
        print(f"Confidence std: {np.std([r['probability'] for r in self.results]):.3f}")

# --- Main execution ---
if __name__ == "__main__":
    print("🎵 Ripeness Classifier Analysis Tool")
    print("="*50)
    
    # Initialize analyzer
    analyzer = RipenessAnalyzer()
    
    # Analyze samples
    analyzer.analyze_samples(SAMPLES_DIR)
    
    # Calculate metrics
    metrics = analyzer.calculate_metrics()
    
    # Create visualizations
    analyzer.plot_results(metrics)
    
    # Save results
    analyzer.save_detailed_results()
    
    print(f"\n🎉 Analysis complete! Check the generated files:")
    print("   - analysis_graphs/ folder (individual PNG files)")
    print("   - ripeness_results.csv (detailed results)")
    
    print(f"\n💡 Tips for better accuracy:")
    print("   - Ensure filenames follow 'ripe*.wav' or 'unripe*.wav' pattern")
    print("   - Adjust THRESHOLD if needed based on ROC curve")
    print("   - Check samples with low confidence scores")
    print("   - Consider retraining with more balanced data if accuracy is low")