# 🎙️ Accent Classifier – KMeans-Based Clustering Tool for English Accents

This project is an attempted solution to **analyze and classify English-speaking accents** from audio input using **MFCC-based feature extraction and KMeans clustering**. It supports both uploaded `.mp3`/`.wav` files and direct links to **YouTube videos**.

Built with:
- 🧠 `scikit-learn` for clustering  
- 🎧 `librosa` for audio processing  
- 🌐 `Streamlit` for an interactive web UI  
- 🎥 `yt-dlp` for YouTube audio extraction  

---

## 🛠️ Features

- Upload an audio file **(MP3/WAV)** or paste a **YouTube link**
- Auto-downloads and extracts audio using `yt-dlp`
- Extracts **MFCC features** for accent analysis
- Predicts the speaker’s **accent cluster** using a pre-trained KMeans model
- Displays **similar samples** from the same cluster
- CLI tools for training and testing models
- Streamlit app for global access and demo

---

## 📁 Project Structure
```
accent-ai-agent/
│
├── app/
│   └── app.py                   # Streamlit UI
|   └── test_model.py            # CLI model testing script
|   └── train_model.py           # CLI model training script
│
├── data/
│   └── samples/                 # Training audio files
│
├── downloads/                   # Temporary folder for YouTube MP3s
├── audio/                       # Temporary folder for uploaded MP3s
│
├── accent_kmeans_model.joblib   # Saved trained KMeans model
├── accent_cluster_reference.csv # Reference of cluster assignments
│
├── requirements.txt
└── README.md
```

## 🚀 Quick Start - How to use this app

### 1. Clone the repo

```
bash
git clone https://github.com/YOUR_USERNAME/accent-ai-agent.git
cd accent-ai-agent
```

### 2. Setup your environment
```
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. 🌐 Run the Web App (Streamlit)
```
streamlit run app/app.py
```

## 🧠 Training Your Own Model
Place your .mp3 or .wav files in data/samples/
Run the trainer using the CLI command
```
python app/train_model.py --n_clusters 8
```
This generates:
accent_kmeans_model.joblib – the trained model
accent_cluster_reference.csv – cluster assignments for training files

## 🔍 Test From Command Line
```
python app/test_model.py --test_file path/to/test_audio.mp3
```
The script:
Loads the trained model
Extracts features from your test audio
Predicts the cluster
Displays similar audio samples from the training set


