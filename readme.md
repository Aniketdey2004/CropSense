# 🌾 CropSense — AI-Powered Crop Rotation Advisor

> **Precision agriculture meets machine learning.** CropSense uses a two-stage KNN + Random Forest pipeline to recommend the best next crop to plant, factoring in your soil conditions, climate, and the agronomic impact of your previous crop.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📸 App Preview

The app features a dark-green sidebar for inputs and a card-based results layout with medal rankings, score bars, and agronomic tags — all running in a Streamlit interface.

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [ML Pipeline Architecture](#-ml-pipeline-architecture)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Crop Metadata Schema](#-crop-metadata-schema)
- [How the Scoring Works](#-how-the-scoring-works)
- [Agronomic Rules](#-agronomic-rules)
- [Installation & Local Setup](#-installation--local-setup)
- [Usage](#-usage)
- [Model Training (Colab)](#-model-training-colab)
- [Model Performance](#-model-performance)
- [Input Parameters](#-input-parameters)
- [Output Explained](#-output-explained)
- [Tech Stack](#-tech-stack)
- [Limitations & Future Work](#-limitations--future-work)

---

## 🌱 Project Overview

Poor crop rotation is one of the leading causes of soil degradation, reduced yields, and excessive fertilizer use in smallholder farming. CropSense addresses this by combining:

- **Environmental matching** — finding crops that historically thrive in your exact soil and climate conditions using K-Nearest Neighbors on 2,200 historical field records.
- **Agronomic ranking** — scoring each candidate crop on rotation soundness, nitrogen restoration, soil depletion impact, water efficiency, and profitability using a Random Forest classifier trained on FAO and ICAR guidelines.

The result is a ranked shortlist of the best crops to plant this season, complete with tags explaining *why* each crop is or isn't a good choice.

---

## ✨ Features

- **Two-stage ML pipeline** — KNN retrieval followed by RF ranking, not a single classifier
- **Agronomic intelligence** — understands heavy feeders, nitrogen fixers, and monoculture risks
- **Perennial exclusion** — automatically filters out multi-year fruit trees (mango, banana, coconut, etc.) from seasonal rotation candidates
- **Environmental match threshold** — crops with fewer than 4% historical presence in similar conditions are excluded
- **Transparent scoring** — every recommendation shows Final Score, AI Confidence, and Environmental Match separately
- **Contextual tags** — 🌿 Restores Nitrogen, ✅ Ideal Rotation, 💧 Low Water, 💰 High Profit, ☀️ Drought Tolerant, ⚠️ Depletes Soil, ⚠️ High Fertilizer, 🚫 Monoculture
- **Crops to avoid** — surfaces sub-optimal candidates with explanations, not just the best picks
- **Beautiful Streamlit UI** — medal cards, score bars, mini stats, and a full comparison table

---

## 🤖 ML Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INPUTS                          │
│   N, P, K, Temperature, Humidity, pH, Rainfall,            │
│   Previous Crop                                             │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│               STAGE 1 — KNN ENVIRONMENTAL RETRIEVAL         │
│                                                             │
│  • StandardScaler normalises the 7 soil/climate features    │
│  • KNN (n_neighbors=50) finds the 50 most similar          │
│    historical field records from 2,200 entries              │
│  • Candidate crops are ranked by frequency among            │
│    the 50 neighbors → "environment_match %" score           │
│  • Perennials removed, crops below 4% match filtered out    │
└────────────────────────────┬────────────────────────────────┘
                             │  shortlist of viable crops
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              STAGE 2 — RANDOM FOREST AGRONOMIC RANKER       │
│                                                             │
│  12 engineered features per crop pair:                      │
│  • previous_soil_depletion, candidate_soil_depletion        │
│  • candidate_nitrogen_fixer, prev_is_heavy_feeder           │
│  • cand_is_heavy_feeder, candidate_market_price             │
│  • candidate_profitability_score, nitrogen_restoration      │
│  • depletion_delta, water_score, fertilizer_score           │
│  • drought_score                                            │
│                                                             │
│  RF outputs P(optimal rotation) → "ai_confidence"          │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                     FINAL SCORING                           │
│                                                             │
│   Final Score = (RF confidence × 0.70)                     │
│               + (env_match / 100 × 0.30) × 100             │
│                                                             │
│   Results sorted descending, top 5 shown as recommendations │
│   Crops with Final Score < 20 shown as "Crops to Avoid"     │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

### Primary Dataset — Crop Recommendation Dataset
- **Source:** [Kaggle — Atharva Ingle's Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- **Size:** 2,200 rows × 8 columns
- **Coverage:** 22 crops, 100 records each
- **Features:** N, P, K (soil nutrients), temperature (°C), humidity (%), pH, rainfall (mm), label (crop name)
- **Quality:** Zero missing values, zero duplicates

### Secondary Dataset — Crop Metadata (Custom)
Hand-crafted agronomic metadata for all 22 crops, stored as `crop_metadata.csv`:

| Column | Type | Description |
|---|---|---|
| `crop` | string | Crop name (matches `label` in primary dataset) |
| `nitrogen_fixer` | int (0/1) | Whether the crop fixes atmospheric nitrogen |
| `water_requirement` | Low/Medium/High | Irrigation demand |
| `market_price_usd_per_ton` | float | Approximate global market price |
| `growing_season_days` | int | Typical days to harvest |
| `soil_depletion_index` | int (1–5) | How much the crop exhausts the soil |
| `drought_tolerance` | Low/Medium/High | Resilience to water scarcity |
| `fertilizer_dependency` | Low/Medium/High | Reliance on synthetic fertilizers |
| `profitability_score` | float (1–10) | Composite profitability rating |

### Training Dataset — Rotation Pairs (Synthetically Labelled)
- **Size:** 484 crop-pair combinations (22 previous × 22 candidate crops)
- **Label:** `optimal` (0 or 1) — assigned using agronomic rules from FAO wheat-legume guides and ICAR rice-based cropping system publications
- **Labelling reasons:** Monoculture, Back-to-back heavy feeders, Legume after heavy feeder, Diverse rotation

---

## 📁 Project Structure

```
CROP_ROTATION/
│
├── data/
│   ├── crop_metadata.csv          # Custom agronomic metadata (22 crops × 9 fields)
│   ├── Crop_recommendation.csv    # Primary Kaggle dataset (2,200 rows)
│   └── master_df.pkl              # Merged & pickled master dataframe (used by KNN)
│
├── models/
│   ├── scaler.pkl                 # Fitted StandardScaler for the 7 soil/climate features
│   ├── knn_model.pkl              # Trained KNN retrieval model (n_neighbors=50)
│   └── rf_ranker.pkl              # Trained Random Forest agronomic ranker
│
├── src/
│   └── pipeline.py                # Core ML inference pipeline (recommend_crop function)
│
├── app.py                         # Streamlit frontend
├── requirements.txt               # Python dependencies
├── .gitignore
└── readme.md
```

---

## 🌿 Crop Metadata Schema

The 22 crops in the system, classified by agronomic role:

**Heavy Feeders** (high nitrogen consumption, soil-depleting):
`rice`, `maize`, `cotton`, `jute`, `sugarcane`

**Nitrogen Fixers** (legumes that restore soil nitrogen):
`chickpea`, `kidneybeans`, `lentil`, `blackgram`, `mungbean`, `mothbeans`, `pigeonpeas`

**Perennials** (excluded from seasonal rotation — multi-year trees):
`mango`, `papaya`, `banana`, `coconut`, `apple`, `orange`, `grapes`, `pomegranate`, `coffee`

**Remaining Seasonal Crops:**
`watermelon`, `muskmelon`, `cotton`, `jute`, `coffee` (non-perennial classification)

---

## 📐 How the Scoring Works

### Stage 1 — Environmental Match
The KNN model is fitted on the 7 scaled soil/climate features from the 2,200 historical records. For a new input, it retrieves the 50 nearest neighbors and counts how many of each crop appear among them.

```
env_match % = (frequency_in_50_neighbors / 50) × 100
```

A crop needs at least 4% (≥ 2 appearances) to pass the threshold — below this, there is insufficient historical evidence.

### Stage 2 — RF Agronomic Confidence
For each candidate, 12 features are engineered from the previous crop and the candidate crop's metadata. The trained Random Forest outputs `P(class=1)` — the probability that this is an optimal rotation.

```
ai_confidence % = rf_model.predict_proba(features)[0][1] × 100
```

### Final Score
```
final_score = (ai_confidence/100 × 0.70 + env_match/100 × 0.30) × 100
```

The 70/30 split weights agronomic soundness more heavily than environmental match alone.

---

## 🚜 Agronomic Rules

The Random Forest training labels were assigned using the following logic, grounded in FAO and ICAR publications:

| Scenario | Label | Reason |
|---|---|---|
| Previous crop = candidate crop | 0 | Monoculture — exhausts soil, increases disease pressure |
| Heavy feeder → Heavy feeder | 0 | Back-to-back depletion, no recovery |
| Heavy feeder → Nitrogen fixer | 1 | Ideal rotation — legume restores nitrogen lost by previous crop |
| Standard → Diverse (different family) | 1 | Diverse rotation — breaks pest cycles |
| Nitrogen fixer → Low-depletion crop | 1 | Capitalises on restored soil |

---

## 🛠 Installation & Local Setup

### Prerequisites
- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/crop-rotation.git
cd crop-rotation

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

> **Note:** The `models/` and `data/` directories must be present with the pre-trained pickle files. These are generated by the Colab training notebook — see the [Model Training](#-model-training-colab) section.

---

## 🖥 Usage

1. **Set soil readings** in the sidebar — N, P, K, Temperature, Humidity, Soil pH, Rainfall
2. **Select your previous crop** from the dropdown
3. Click **🔍 Analyse & Recommend**
4. Review:
   - **Field Snapshot** — confirms your inputs at a glance
   - **Top Recommendations** — up to 5 medal-ranked cards with scores and tags
   - **Crops to Avoid** — flagged candidates with reasons
   - **Complete Rankings** — full sortable table with all metrics

---

## 🔬 Model Training (Colab)

The complete training pipeline is available in the Colab notebook:
**[Open in Google Colab](https://colab.research.google.com/drive/1hu4n7RItFPC9PHnSV4JMyKeZ3ebZLrmz?usp=sharing)**

The notebook is structured across four phases:

### Phase 1 — Exploratory Data Analysis
- Loads both `Crop_recommendation.csv` and `crop_metadata.csv`
- Inspects shapes, missing values, duplicates
- Merges into `master_df` (2,200 rows × 17 columns)
- Generates distribution plots, correlation heatmaps, and descriptive statistics

### Phase 2 — KNN Retrieval Model
- Scales the 7 input features with `StandardScaler`
- Fits `NearestNeighbors(n_neighbors=50, algorithm='auto', metric='euclidean')` on all 2,200 records
- Saves `scaler.pkl`, `knn_model.pkl`, and `master_df.pkl`
- Tests retrieval on a low-nitrogen maize-exhausted scenario

### Phase 3 — Random Forest Ranker Training
- Generates all 484 crop-pair combinations (22 × 22)
- Engineers 12 features per pair from `crop_metadata.csv`
- Labels each pair as `optimal=1` or `optimal=0` using FAO/ICAR agronomic rules
- Trains `RandomForestClassifier` with 5-fold cross-validation
- Saves `rf_ranker.pkl`

### Phase 4 — End-to-End System Test
- Combines both models into the full `recommend_crop()` pipeline
- Tests on a `maize` → recommendations scenario
- Downloads all three model artifacts for use in the Streamlit app

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| Cross-Validation F1 Score | **0.920 ± 0.022** |
| CV Strategy | 5-Fold Stratified |
| Training Set Size | 484 crop-pair combinations |
| Classes | Optimal (1) / Suboptimal (0) |
| RF Features | 12 engineered agronomic features |

---

## 🎛 Input Parameters

| Parameter | Range | Unit | Description |
|---|---|---|---|
| Nitrogen (N) | 0 – 140 | kg/ha | Soil nitrogen content |
| Phosphorus (P) | 5 – 145 | kg/ha | Soil phosphorus content |
| Potassium (K) | 5 – 205 | kg/ha | Soil potassium content |
| Temperature | 10 – 45 | °C | Average ambient temperature |
| Humidity | 20 – 100 | % | Relative humidity |
| Soil pH | 3.0 – 10.0 | — | Soil acidity/alkalinity |
| Rainfall | 20 – 300 | mm | Expected rainfall |
| Previous Crop | dropdown | — | One of 22 crops in the dataset |

---

## 📋 Output Explained

Each recommended crop returns the following fields:

| Field | Description |
|---|---|
| `final_score` | Composite score (0–100%): 70% RF confidence + 30% env match |
| `ai_confidence` | Random Forest probability of optimal rotation (0–100%) |
| `env_match` | % of the 50 KNN neighbors where this crop was grown |
| `market_price` | Global market price in USD/ton |
| `profitability` | Profitability score out of 10 |
| `water_req` | Low / Medium / High water requirement |
| `nitrogen_fixer` | 1 if the crop restores nitrogen, else 0 |
| `soil_depletion` | Soil depletion index 1 (low) to 5 (high) |
| `tags` | Contextual agronomic labels (good / warn / bad) |

---

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| ML Models | scikit-learn (KNN, Random Forest) |
| Data Processing | pandas, NumPy |
| Model Serialisation | joblib |
| Training Environment | Google Colab |
| Styling | Custom CSS injected via `st.markdown` |
| Fonts | DM Sans, Space Grotesk (Google Fonts) |

### `requirements.txt`
```
streamlit
pandas
numpy
scikit-learn
joblib
```

---


## 📚 References

- FAO Wheat-Legume Rotation Guidelines — Food and Agriculture Organization of the United Nations
- ICAR Rice-Based Cropping System Publications — Indian Council of Agricultural Research
- Ingle, A. (2020). *Crop Recommendation Dataset*. Kaggle. https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

---

## 🙏 Acknowledgements

- Dataset by [Atharva Ingle](https://www.kaggle.com/atharvaingle) on Kaggle
- Agronomic classification based on publicly available FAO and ICAR crop rotation literature
- UI design inspired by precision agriculture dashboards

---

*Built with 🌱 for sustainable agriculture*