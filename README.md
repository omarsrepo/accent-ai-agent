# 🎙️ Accent Classifier – KMeans-Based Clustering Tool for English Accents

Analyze and classify English-speaking accents from audio input using pre-trained machine learning models from huggingface and proxy methods to download audio source (video URL). It supports both uploaded `.mp3`/`.wav` files and direct links to **YouTube videos**.  

The model currently classifies accents of a provided mp3 file by clustering it into a specified number of clusters. The model was trained on a small sample size of 20 english audio files with 5 different accents across them. It is just meant to simulate supervised learning as I could not obtain mp3 files along with tsv/xlsx files that contained metadata about each mp3 file to train the model on a better method such as SVM, RandomForest or Neural Networks, which i believe would improve the accuracy and reliability of this application much more. This is just meant to be a proof of concept.  
The streamlit app is simple to use as you simply provide a url to a video, the audio file will be extracted and then placed into a cluster. You will also be shown some similar audio files that belong to the assigned cluster to give you an idea of which accent it was clustered along with. This might not be accurate but once again, its just a proof of concept!

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

---

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

---

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


