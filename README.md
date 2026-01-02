


# 🎨 Moodify - ML Hybrid Music Recommender System

A sophisticated mood-based music recommendation system that combines natural language processing, unsupervised learning, and content-based filtering to deliver personalized song recommendations based on your emotional state.

##  Problem Description
Music streaming platforms often struggle to balance **emotional relevance**, **personal taste**, and **explainability** in recommendation systems. Many systems rely purely on collaborative filtering, which fails for new users (cold start) and provides little transparency.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Machine Learning Components](#machine-learning-components)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Data](#data)

## 🎯 Overview

Moodify is an intelligent music recommendation system that understands your mood through natural language input and recommends songs that match your emotional state. The system learns from your preferences over time, building a personalized taste profile that improves recommendations with each saved song.

## ✨ Features

- **🎭 Mood Detection**: Choose from 4 moods (Happy, Sad, Calm, Energetic) or describe your feelings in natural language
- **🧠 NLP Classification**: Uses Logistic Regression to classify mood from text input
- **🎵 Smart Recommendations**: Hybrid recommendation system combining:
  - Content-based filtering using Euclidean distance
  - K-Means clustering for song grouping
  - Personalized taste profile matching
- **📊 Personal Palette**: Build and analyze your musical taste profile
- **🔍 ML Inspector**: Transparent view into the machine learning algorithms powering recommendations
- **🎨 Beautiful UI**: Modern, gradient-based interface with interactive visualizations

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Machine Learning**: scikit-learn
  - Logistic Regression (text classification)
  - K-Means Clustering (unsupervised learning)
  - StandardScaler (feature normalization)
- **Data Processing**: pandas, numpy
- **Visualization**: Plotly Express
- **Data Storage**: JSON (local file-based)

## 🤖 Machine Learning Components

### 1. **NLP Mood Classification (Logistic Regression)**
- Classifies user text input into one of four mood categories
- Uses CountVectorizer for text feature extraction
- Provides confidence scores for each mood category

### 2. **K-Means Clustering**
- Groups songs into 8 distinct clusters based on audio features
- Features used: energy, valence, tempo, danceability, acousticness, instrumentalness
- Helps identify musical neighborhoods and similar tracks

### 3. **Mood Assignment Algorithm**
- Rule-based system that assigns moods to songs based on audio features:
  - **Sad**: Low valence (< 0.25), low energy (< 0.3), minor mode
  - **Energetic**: High energy (> 0.75), high tempo (> 120 BPM)
  - **Happy**: High valence (> 0.6), major mode
  - **Calm**: Default for other tracks

### 4. **Content-Based Filtering**
- Uses Euclidean distance to find songs similar to user's saved preferences
- Builds a user profile from saved tracks' audio features
- Recommends songs closest to the user's taste profile

## 📦 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Steps

1. **Clone the repository** (or navigate to the project directory):
   ```
   cd MOODIFY
   ```

2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Ensure data files are present**:
   - `high_popularity_spotify_data.csv`
   - `low_popularity_spotify_data.csv`

## 🚀 Usage

1. **Start the Streamlit application**:
   ```
   streamlit run app.py
   ```

2. **Open your browser** to the URL shown in the terminal (typically `http://localhost:8501`)

3. **Explore the application**:
   - **Explore Moods Tab**: Select a mood or type how you're feeling to get song recommendations
   - **Personal Palette Tab**: View your saved songs and taste profile analysis
   - **Under the Hood Tab**: Explore the ML algorithms and data visualizations

### Quick Start Guide

1. **Choose your mood**: Click one of the mood buttons (Happy, Sad, Calm, Energetic) or type a description
2. **Browse recommendations**: View 4 songs that match your mood
3. **Save favorites**: Click "❤️ Save to Palette" on songs you like
4. **Build your profile**: The more songs you save, the better recommendations become
5. **Explore**: Use the "🔄 New Songs" button to get fresh recommendations

## 📁 Project Structure

```
MOODIFY/
├── images                          # These images are for the Model results

├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── Data Files:
├── high_popularity_spotify_data.csv    # High popularity tracks dataset
├── low_popularity_spotify_data.csv     # Low popularity tracks dataset
│
├── Model Files (if saved):
├── kmeans.joblib                   # Trained K-Means model (optional)
├── mood_text_clf.joblib            # Text classifier model (optional)
├── scaler.joblib                   # Feature scaler (optional)
│
└── moodify_v4.json                # User saved songs database
├── moodify-project-presentation.pdf  # this is our project's presentation

```

## 🔬 How It Works

### Recommendation Pipeline

1. **User Input**: User selects a mood or types a text description
2. **Mood Classification**: If text is provided, Logistic Regression classifies the mood
3. **Song Pooling**: System filters songs matching the detected mood
4. **Personalization**:
   - If user has saved songs: Calculate user profile and find similar tracks using Euclidean distance
   - If no saved songs: Randomly sample from mood-matched pool
5. **Display**: Show 4 recommended songs with Spotify embeds

### Feature Engineering

The system uses the following Spotify audio features:
- **Energy**: Intensity and power (0.0 to 1.0)
- **Valence**: Musical positiveness (0.0 to 1.0)
- **Tempo**: Beats per minute
- **Danceability**: How suitable a track is for dancing (0.0 to 1.0)
- **Acousticness**: Confidence measure of whether the track is acoustic (0.0 to 1.0)
- **Instrumentalness**: Predicts whether a track contains no vocals (0.0 to 1.0)

All features are normalized using StandardScaler before clustering and similarity calculations.

## 📊 Data

The application uses two Spotify datasets:
- **High Popularity Dataset**: Popular tracks from Spotify playlists
- **Low Popularity Dataset**: Less popular tracks to provide diverse recommendations

Each dataset contains:
- Track metadata (name, artist, album)
- Audio features (energy, valence, tempo, etc.)
- Spotify track IDs for embedding
- Playlist information

## 🎨 UI Features

- **Dark Theme**: Purple gradient background with elegant typography
- **Interactive Visualizations**: 
  - Radar charts for mood fingerprints
  - Bar charts for classification confidence
  - Pie charts for cluster distribution
  - Heatmaps for cluster signatures
- **Spotify Integration**: Embedded Spotify players for each recommended track
- **Real-time Updates**: Dynamic recommendations that update based on user interactions

## 🔧 Configuration

The application uses a JSON file (`moodify_v4.json`) to persist user's saved songs. This file is automatically created and updated when users save tracks.

## 📝 Notes

- The ML models are trained on-the-fly when the application starts
- K-Means uses 8 clusters with a random state of 42 for reproducibility
- The text classifier is trained on a small curated dataset of mood-related keywords
- All audio features are scaled before clustering to ensure equal weight

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

## 📄 License

This project is open source and available for educational purposes.

---

**Made with ❤️ using Streamlit, scikit-learn, and Spotify data**

