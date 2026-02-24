# 🎵 VibeMatch AI: Dual-Model Music Discovery Engine

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


