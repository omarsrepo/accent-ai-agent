import os
import librosa
import numpy as np
import pandas as pd
import argparse
from joblib import load

# Constants
SAMPLE_RATE = 16000
N_MFCC = 13
MODEL_PATH = 'accent_kmeans_model.joblib'
REFERENCE_CSV = 'accent_cluster_reference.csv'

def extract_features(audio_path):
    """Extract MFCCs from an audio file."""
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean.astype('float64')

def load_model_and_reference():
    """Load the trained KMeans model and reference CSV."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"[!] KMeans model not found at: {MODEL_PATH}")
    if not os.path.exists(REFERENCE_CSV):
        raise FileNotFoundError(f"[!] Reference CSV not found at: {REFERENCE_CSV}")
    
    print("[+] Loading trained model...")
    model = load(MODEL_PATH)

    print("[+] Loading reference data...")
    reference_df = pd.read_csv(REFERENCE_CSV)

    return model, reference_df

def predict_cluster(test_file_path, model, reference_df):
    """Predict the cluster of the test file and show similar samples."""
    print(f"[+] Extracting features for test file: {test_file_path}")
    feature = extract_features(test_file_path)

    cluster = model.predict([feature])[0]
    print(f"[✓] Predicted Cluster: {cluster}")

    # Find matching files from same cluster
    print("\n[+] Files from the same cluster:")
    matches = reference_df[reference_df['cluster'] == cluster]
    for i, row in matches.head(10).iterrows():
        print(f" - {row['filename']}")

    return cluster, matches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, required=True, help='Path to test MP3 file')
    args = parser.parse_args()

    model, ref_df = load_model_and_reference()

    predict_cluster(args.test_file, model, ref_df)

if __name__ == '__main__':
    main()
