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

