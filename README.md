# 🎵 VibeMatch AI: Dual-Model Music Discovery Engine
[https://recommendation-engine-hvgabpdv76g4cl3wrysqjm.streamlit.app/](url)

## 📌 The Problem
Traditional music recommendation systems often focus strictly on "Vibe" (sonic similarity) or "Popularity" (current charts). This creates a gap where a user might find a similar-sounding song that lacks the structural "DNA" of a successful track. **VibeMatch AI** bridges this gap by combining **Unsupervised Learning** for discovery and **Supervised Learning** for commercial risk assessment.

## 💡 The Solution
This application employs a dual-layered AI architecture:

1.  **Similarity Engine (KNN):** Maps songs in a multi-dimensional space to identify the 10 most sonically similar tracks based on audio DNA (Acousticness, Danceability, etc.).
2.  **Hit Predictor (XGBoost):** Passes those recommendations through a classification model to determine if they possess the structural characteristics of a "Hit," trained on a historical dataset of 56,000 tracks.



---

## 🚀 Key Features
* **Acoustic Fingerprinting:** Utilizes **K-Nearest Neighbors** to map songs based on high-dimensional audio features.
* **Predictive Hit Analysis:** An **XGBoost classifier** evaluates recommendations to provide a "Hit Potential" score.
* **Dynamic Feature Engineering:** Includes custom-engineered features like *Artist Presence* and *Energy-to-Acoustic Ratios* to capture non-linear market trends.
* **Interactive UI:** A custom **Streamlit dashboard** allowing users to search, match, and analyze tracks in real-time.

---

## 🧠 Technical Deep Dive

### **Optimization & Tuning**
To address a **75/25 class imbalance** (Non-Hits vs. Hits), the following strategies were implemented:
* **Scale_Pos_Weight:** Configured to penalize the model more heavily for "False Negatives," ensuring high-potential tracks aren't overlooked.
* **GridSearchCV:** Systematic hyperparameter tuning of `max_depth`, `learning_rate`, and `subsample` to find the optimal global minimum.
* **Metric Choice:** Prioritized **Recall (0.71)** over Precision. In talent discovery (A&R), the cost of missing a "Hit" is significantly higher than the cost of vetting a "False Positive."



### **Feature Importance**
Through model interpretation, it was determined that **Artist Presence** and **Energy** were the primary drivers of commercial success, whereas features like "Positivity" (Valence) had a negligible impact on a song's hit probability in the current market.

## 🛠️ Data Engineering & Pipeline

### **BigQuery Integration & Scalability**
Before model training, the raw dataset was processed using **Google BigQuery**. This allowed for efficient handling of 56,000+ records and ensured data integrity through SQL-based transformation.

* **Data Cleaning:** Leveraged SQL to filter out null values and duplicates, ensuring the model trained on high-quality, unique track metadata.
* **Feature Scaling:** Implemented **Min-Max Scaling** directly within the query to normalize features like `popularity`, `tempo`, and `loudness`. By scaling at the database level, I reduced the computational load on the local Python environment.
* **Feature Engineering:** Initial feature ratios were calculated using SQL window functions to prepare the schema for the KNN and XGBoost models.

### **The Extraction Query**
Below is the optimized SQL query used to pull and preprocess the data from the cloud:

```sql
-- Query to clean, scale, and extract music metadata
CREATE OR REPLACE TABLE `recommendation-engine-488319.spotify.cleaned_spotify_data` AS
WITH base_data AS (
  SELECT 
    track_id,
    track_name,
    artists,
    track_genre,
    popularity,
    danceability,
    energy,
    speechiness,
    acousticness,
    instrumentalness,
    valence,
    tempo,
    loudness
  FROM `recommendation-engine-488319.spotify.raw_spotify_data`
  WHERE track_name IS NOT NULL 
    AND artists IS NOT NULL
),
deduplicated AS (
  -- Keeps only the most popular version of a song to avoid redundant recommendations
  SELECT * FROM base_data
  QUALIFY ROW_NUMBER() OVER(PARTITION BY track_name, artists ORDER BY popularity DESC) = 1
)
SELECT 
    track_id,
    track_name,
    artists,
    track_genre,
    -- Core Features (already 0-1)
    danceability,
    energy,
    speechiness,
    acousticness,
    instrumentalness,
    valence,
    -- Normalized Features (scaling based on standard Spotify ranges)
    (popularity - 0) / (100 - 0) AS norm_popularity,
    (tempo - 0) / (250 - 0) AS norm_tempo,
    (loudness - (-60)) / (0 - (-60)) AS norm_loudness
FROM deduplicated;


