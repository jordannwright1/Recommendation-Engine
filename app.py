import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page config
st.set_page_config(page_title="VibeMatch AI + HitPredictor", page_icon="🎵", layout="wide")

# 1. Load Assets
@st.cache_resource
def load_assets():
    # Recommendation Model (KNN)
    knn_model = joblib.load('music_model.joblib')
    # Classification Model (XGBoost)
    hit_model = joblib.load('hit_predictor_model.joblib')
    # The feature list we saved from our XGBoost notebook
    hit_features = joblib.load('model_features.joblib')
    # The dataframe
    df = pd.read_pickle('music_data.pkl')
    return knn_model, hit_model, hit_features, df

knn_model, hit_model, hit_features, df = load_assets()

# 2. UI Header
st.title("🎵 VibeMatch AI")
st.markdown("Find your next favorite song.")

# 3. User Input
selected_song = st.selectbox("Search for a song:", options=df['track_name'].unique())

if st.button('Recommend & Analyze'):
    # Find index
    song_idx = df[df['track_name'] == selected_song].index[0]
    
    # --- STEP 1: KNN RECOMMENDATION ---
    knn_cols = ['danceability', 'energy', 'speechiness', 'acousticness', 
                'instrumentalness', 'valence', 'norm_popularity', 
                'norm_tempo', 'norm_loudness']
    
    query_knn = df.iloc[song_idx][knn_cols].values.reshape(1, -1)
    distances, indices = knn_model.kneighbors(query_knn, n_neighbors=11)
    
    st.subheader(f"Recommendations based on '{selected_song}':")
    st.divider()

    # --- STEP 2: LOOP THROUGH RESULTS & PREDICT HITS ---
    for i in range(1, len(indices[0])):
        res_idx = indices[0][i]
        rec_row = df.iloc[res_idx]
        
        # Prepare features for XGBoost 
        query_hit = rec_row[hit_features].values.reshape(1, -1)
        # Get probability of being a '1' (Hit)
        hit_prob = hit_model.predict_proba(query_hit)[0][1] 
        
        # UI Display
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.write(f"**{rec_row['track_name']}**")
            st.caption(f"by {rec_row['artists']}")
        
        with col2:
            match_pct = (1 - distances[0][i]) * 100
            st.metric("Vibe Match", f"{match_pct:.1f}%")
            
        with col3:
            # Color code based on probability
            if hit_prob > 0.7:
                label = "🔥 High"
            elif hit_prob > 0.55:
                label = "📈 Moderate"
            else:
                label = "🧊 Low"
            
            st.metric("Hit Potential", label, delta=f"{hit_prob*100:.1f}%")
        
        st.divider()
