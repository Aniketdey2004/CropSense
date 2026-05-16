import pandas as pd
import numpy as np
import joblib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

scaler    = joblib.load(os.path.join(ROOT, 'models', 'scaler.pkl'))
knn_model = joblib.load(os.path.join(ROOT, 'models', 'knn_model.pkl'))
rf_model  = joblib.load(os.path.join(ROOT, 'models', 'rf_ranker.pkl'))
master_df = pd.read_pickle(os.path.join(ROOT, 'data', 'master_df.pkl'))
meta_df   = pd.read_csv(os.path.join(ROOT, 'data', 'crop_metadata.csv'))

crop_info = {}
for _, row in meta_df.iterrows():
    crop_info[row['crop']] = {
        'nitrogen_fixer':        row['nitrogen_fixer'],
        'water_requirement':     row['water_requirement'],
        'market_price':          row['market_price_usd_per_ton'],
        'soil_depletion':        row['soil_depletion_index'],
        'fertilizer_dependency': row['fertilizer_dependency'],
        'drought_tolerance':     row['drought_tolerance'],
        'profitability_score':   row['profitability_score']
    }

water_score_map      = {'Low': 3, 'Medium': 2, 'High': 1}
fertilizer_score_map = {'Low': 3, 'Medium': 2, 'High': 1}
drought_score_map    = {'High': 3, 'Medium': 2, 'Low': 1}

heavy_feeders = ['rice', 'maize', 'cotton', 'jute', 'sugarcane']
nitrogen_fixers = [
    'chickpea', 'kidneybeans', 'lentil', 'blackgram',
    'mungbean', 'mothbeans', 'pigeonpeas'
]
# Perennial crops — multi-year trees, not valid seasonal rotation candidates
# Source: basic agronomic classification
perennials = [
    'mango', 'papaya', 'banana', 'coconut',
    'apple', 'orange', 'grapes', 'pomegranate', 'coffee'
]

RF_WEIGHT  = 0.7
ENV_WEIGHT = 0.3

FEATURE_COLUMNS = [
    'previous_soil_depletion', 'candidate_soil_depletion',
    'candidate_nitrogen_fixer', 'prev_is_heavy_feeder',
    'cand_is_heavy_feeder', 'candidate_market_price',
    'candidate_profitability_score', 'nitrogen_restoration',
    'depletion_delta', 'water_score', 'fertilizer_score', 'drought_score'
]


def retrieve_candidates(N, P, K, temperature, humidity, ph, rainfall, top_k=50):
    input_vector = pd.DataFrame([{
        'N': N, 'P': P, 'K': K,
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall
    }])
    scaled     = scaler.transform(input_vector)
    _, indices = knn_model.kneighbors(scaled, n_neighbors=top_k)
    neighbors  = master_df.iloc[indices[0]]
    counts     = neighbors['label'].value_counts().reset_index()
    counts.columns = ['crop', 'frequency']
    counts['environment_match'] = (counts['frequency'] / top_k) * 100
    return counts


def generate_candidate_features(previous_crop, candidate_crop):
    prev = crop_info[previous_crop]
    cand = crop_info[candidate_crop]
    nitrogen_restoration = (
        1 if previous_crop in heavy_feeders and cand['nitrogen_fixer'] == 1
        else 0
    )
    return {
        'previous_soil_depletion':       prev['soil_depletion'],
        'candidate_soil_depletion':      cand['soil_depletion'],
        'candidate_nitrogen_fixer':      cand['nitrogen_fixer'],
        'prev_is_heavy_feeder':          1 if previous_crop in heavy_feeders else 0,
        'cand_is_heavy_feeder':          1 if candidate_crop in heavy_feeders else 0,
        'candidate_market_price':        cand['market_price'],
        'candidate_profitability_score': cand['profitability_score'],
        'nitrogen_restoration':          nitrogen_restoration,
        'depletion_delta':               prev['soil_depletion'] - cand['soil_depletion'],
        'water_score':                   water_score_map[cand['water_requirement']],
        'fertilizer_score':              fertilizer_score_map[cand['fertilizer_dependency']],
        'drought_score':                 drought_score_map[cand['drought_tolerance']]
    }


def recommend_crop(previous_crop, N, P, K,
                   temperature, humidity, ph, rainfall, top_k=50):

    candidates = retrieve_candidates(N, P, K, temperature, humidity, ph, rainfall, top_k)

    # Fix 1 — Remove perennial fruit trees
    # These are multi-year crops, not valid seasonal rotation candidates
    candidates = candidates[~candidates['crop'].isin(perennials)]

    # Fix 2 — Minimum environmental match threshold
    # A crop with <10% match has fewer than 5 historical precedents
    # in similar soil conditions — not enough evidence to recommend
    candidates = candidates[candidates['environment_match'] >= 10]

    results = []

    for _, row in candidates.iterrows():
        crop      = row['crop']
        env_match = row['environment_match']
        feat_row  = generate_candidate_features(previous_crop, crop)
        feat_df   = pd.DataFrame([feat_row])[FEATURE_COLUMNS]
        prob      = rf_model.predict_proba(feat_df)[0][1]
        score     = (prob * RF_WEIGHT + (env_match / 100) * ENV_WEIGHT) * 100
        cand      = crop_info[crop]

        tags = []
        if cand['nitrogen_fixer'] == 1:
            tags.append(('🌿 Restores Nitrogen', 'good'))
        if previous_crop in heavy_feeders and cand['nitrogen_fixer'] == 1:
            tags.append(('✅ Ideal Rotation', 'good'))
        if cand['water_requirement'] == 'Low':
            tags.append(('💧 Low Water', 'good'))
        if cand['profitability_score'] >= 8:
            tags.append(('💰 High Profit', 'good'))
        if cand['drought_tolerance'] == 'High':
            tags.append(('☀️ Drought Tolerant', 'good'))
        if cand['soil_depletion'] >= 4:
            tags.append(('⚠️ Depletes Soil', 'warn'))
        if cand['fertilizer_dependency'] == 'High':
            tags.append(('⚠️ High Fertilizer', 'warn'))
        if previous_crop == crop:
            tags.append(('🚫 Monoculture', 'bad'))

        results.append({
            'crop':           crop,
            'env_match':      round(env_match, 1),
            'ai_confidence':  round(prob * 100, 1),
            'final_score':    round(score, 1),
            'tags':           tags,
            'market_price':   cand['market_price'],
            'profitability':  cand['profitability_score'],
            'water_req':      cand['water_requirement'],
            'nitrogen_fixer': cand['nitrogen_fixer'],
            'soil_depletion': cand['soil_depletion'],
        })

    return (
        pd.DataFrame(results)
        .sort_values('final_score', ascending=False)
        .reset_index(drop=True)
    )