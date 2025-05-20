import os
import librosa
import numpy as np
from sklearn.cluster import KMeans
import argparse

# Configuration
SAMPLE_RATE = 16000
N_MFCC = 13
N_CLUSTERS = 3  # Assuming 3 accents for demo


def extract_features(audio_path):
    """Extract MFCCs from an audio file."""
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return np.mean(mfcc.T, axis=0)  # Average over time


def load_audio_features(audio_dir):
    """Load and process all audio files in a directory."""
    features = []
    filenames = []
    for filename in os.listdir(audio_dir):
        if filename.endswith(".wav") or filename.endswith(".mp3"):
            path = os.path.join(audio_dir, filename)
            mfcc = extract_features(path)
            features.append(mfcc)
            filenames.append(filename)
    return np.array(features), filenames


def cluster_accents(features):
    """Cluster feature vectors into accent groups."""
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels


def predict_accent(file_path, kmeans_model):
    feature = extract_features(file_path)
    label = kmeans_model.predict([feature])[0]
    return label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='data/samples', help='Directory with sample training audios')
    parser.add_argument('--input', type=str, required=True, help='Path to input audio file')
    args = parser.parse_args()

    print("[+] Extracting training features...")
    features, _ = load_audio_features(args.train_dir)
    print("[+] Clustering accents...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42).fit(features)

    print(f"[+] Predicting accent for: {args.input}")
    accent_label = predict_accent(args.input, kmeans)
    print(f"→ Predicted Accent Cluster: {accent_label}")


if __name__ == '__main__':
    main()
