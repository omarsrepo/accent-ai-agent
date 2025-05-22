import os
import librosa
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from joblib import dump
from tqdm import tqdm
import argparse

# === GLOBAL CONFIGURATION ===
TRAIN_DATA_DIR = r"data\samples"
SAMPLE_RATE = 16000
N_MFCC = 13
DEFAULT_N_CLUSTERS = 8
MODEL_OUTPUT = "accent_kmeans_model.joblib"
REFERENCE_CSV = "accent_cluster_reference.csv"


def extract_features(audio_path):
    """Extract MFCC features from an audio file."""
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean.astype('float64')


def load_audio_features(audio_dir):
    """Load all audio files and extract features."""
    features = []
    filenames = []

    for filename in tqdm(os.listdir(audio_dir), desc="Extracting features"):
        if filename.endswith(".mp3") or filename.endswith(".wav"):
            path = os.path.join(audio_dir, filename)
            try:
                mfcc = extract_features(path)
                features.append(mfcc)
                filenames.append(filename)
            except Exception as e:
                print(f"[-] Skipped {filename} due to error: {e}")

    return np.array(features), filenames


def plot_elbow(features, max_k=20):
    """Plot the Elbow method to help choose optimal k."""
    distortions = []
    K_range = range(2, max_k + 1)

    print("[+] Running Elbow method to determine optimal K...")
    for k in tqdm(K_range):
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(features)
        distortions.append(model.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(K_range, distortions, marker='o')
    plt.title("Elbow Method For Optimal K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia (Distortion)")
    plt.grid(True)
    plt.show()


def train_model(features, filenames, n_clusters):
    """Train KMeans and save results."""
    print(f"[+] Clustering into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)

    print(f"[+] Saving model to {MODEL_OUTPUT}")
    dump(kmeans, MODEL_OUTPUT)

    print(f"[+] Saving cluster reference to {REFERENCE_CSV}")
    df = pd.DataFrame({
        'filename': filenames,
        'cluster': labels
    })
    df.to_csv(REFERENCE_CSV, index=False)

    return kmeans, df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--elbow', action='store_true', help='Run Elbow Method plot to determine optimal K')
    parser.add_argument('--n_clusters', type=int, default=DEFAULT_N_CLUSTERS, help='Number of clusters to form')
    parser.add_argument('--data_dir', type=str, default=TRAIN_DATA_DIR, help='Path to training dataset')

    args = parser.parse_args()

    print(f"[+] Loading training data from: {args.data_dir}")
    features, filenames = load_audio_features(args.data_dir)

    if args.elbow:
        plot_elbow(features)

    # Train and save
    train_model(features, filenames, args.n_clusters)


if __name__ == '__main__':
    main()
