import streamlit as st
import os
import uuid
import pandas as pd
import librosa
import numpy as np
from joblib import load
import subprocess
from pathlib import Path
import yt_dlp


# === Configuration ===
SAMPLE_RATE = 16000
N_MFCC = 13
MODEL_PATH = "accent_kmeans_model.joblib"
REFERENCE_CSV = "accent_cluster_reference.csv"
DOWNLOAD_DIR = "downloads"
AUDIO_DIR = "audio"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# === Load Model and Reference Data ===
@st.cache(allow_output_mutation=True)
def load_model_and_data():
    model = load(MODEL_PATH)
    df = pd.read_csv(REFERENCE_CSV)
    return model, df

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean.astype('float64')

def download_audio_from_youtube(url):
    unique_id = str(uuid.uuid4())
    output_path = os.path.join(DOWNLOAD_DIR, f"{unique_id}.mp3")
    command = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "mp3",
        "--no-playlist",
        "-o", output_path,
        url
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

def predict_cluster(file_path, model, ref_df):
    features = extract_features(file_path)
    cluster = model.predict([features])[0]
    matches = ref_df[ref_df['cluster'] == cluster]
    return cluster, matches

# === Streamlit UI ===
st.title("Accent Classification Tool")
model, ref_df = load_model_and_data()

input_option = st.radio("Choose Input Type:", ["Upload Audio File", "YouTube Link"])

if input_option == "Upload Audio File":
    uploaded_file = st.file_uploader("Upload an MP3 or WAV file", type=["mp3", "wav"])
    if uploaded_file:
        temp_path = os.path.join(AUDIO_DIR, f"{uuid.uuid4()}.mp3")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        st.audio(temp_path)
        cluster, matches = predict_cluster(temp_path, model, ref_df)
        st.success(f"Assigned to cluster {cluster}")
        st.subheader("Similar accents in this cluster:")
        st.write(matches.head(10))

elif input_option == "YouTube Link":
    url = st.text_input("Paste YouTube video URL")
    if url and st.button("Process"):
        audio_path = download_audio_from_youtube(url)
        st.audio(audio_path)
        cluster, matches = predict_cluster(audio_path, model, ref_df)
        st.success(f"Assigned to cluster {cluster}")
        st.subheader("Similar accents in this cluster:")
        st.write(matches.head(10))
