🎵 VibeMatch AI: Dual-Model Music Discovery Engine
The Problem
Music recommendation systems often focus solely on "Vibe" (similarity) or "Popularity" (charts). VibeMatch AI bridges this gap by combining Unsupervised Learning for discovery and Supervised Learning for commercial risk assessment.

The Solution
This application uses a dual-layered AI approach:

Similarity Engine (KNN): Finds the 10 most sonically similar tracks based on audio DNA (Acousticness, Danceability, etc.).

Hit Predictor (XGBoost): Analyzes those recommendations to determine if they have the structural characteristics of a "Hit" based on a historical dataset of 56,000 tracks.

🚀 Key Features
Acoustic Fingerprinting: Uses K-Nearest Neighbors to map songs in a multi-dimensional feature space.

Predictive Hit Analysis: An XGBoost classifier evaluates recommendations for "Hit Potential."

Dynamic Feature Engineering: Includes custom-built features like Artist Presence and Energy-to-Acoustic Ratios to boost model accuracy.

Interactive UI: A Streamlit dashboard that allows users to search, match, and analyze tracks in real-time.

🧠 Technical Deep Dive

Optimization & Tuning
Because the dataset had a 75/25 class imbalance (Non-Hits vs. Hits), I utilized:

Scale_Pos_Weight: To penalize the model more for missing a "Hit."

GridSearchCV: To tune hyperparameters (max_depth, learning_rate, subsample).

Metric Choice: Prioritized Recall (0.71) over Precision, ensuring that the model captures as many potential hits as possible—a critical requirement for A&R and talent discovery.

Feature Importance
The model determined that Artist Presence and Energy were the primary drivers of commercial success, while "Positivity" had a negligible impact in the current market.
