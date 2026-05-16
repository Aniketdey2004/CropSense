import streamlit as st
import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from pipeline import recommend_crop, crop_info, heavy_feeders, nitrogen_fixers

st.set_page_config(
    page_title="CropRotate AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header  { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Hide sidebar collapse button entirely ── */
[data-testid="collapsedControl"]        { display: none !important; }
[data-testid="stSidebarCollapseButton"] { display: none !important; }

/* ── Page background ── */
.stApp {
    background:
        radial-gradient(circle at top right, rgba(74,222,128,0.08), transparent 30%),
        radial-gradient(circle at bottom left, rgba(34,197,94,0.05), transparent 30%),
        #f6fbf7;
}

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #14532d 0%, #166534 35%, #15803d 100%);
    border-radius: 24px;
    padding: 2.8rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(21,128,61,0.18);
}
.hero::before {
    content: '';
    position: absolute;
    width: 380px; height: 380px;
    top: -170px; right: -100px;
    background: radial-gradient(circle, rgba(255,255,255,0.12), transparent 70%);
    border-radius: 50%;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    color: #dcfce7;
    padding: 0.3rem 0.85rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.9rem;
    font-weight: 700;
    color: white;
    margin: 0 0 0.5rem 0;
    line-height: 1.08;
}
.hero-title span { color: #86efac; }
.hero-sub {
    color: rgba(255,255,255,0.80);
    font-size: 1.02rem;
    font-weight: 300;
    margin: 0;
    max-width: 560px;
    line-height: 1.7;
}

/* ── Section header ── */
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #0f172a;
    margin: 1.8rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(34,197,94,0.3), transparent);
    margin-left: 0.75rem;
}

/* ── Rank cards ── */
.rank-card {
    border-radius: 20px;
    padding: 1.5rem 1.6rem;
    margin-bottom: 0.5rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 0 2px 12px rgba(15,23,42,0.06);
}
.rank-card:hover { transform: translateY(-3px); box-shadow: 0 10px 30px rgba(34,197,94,0.12); }
.rank-card.gold   { background: linear-gradient(135deg,#fff9db,#fef3c7); border: 1.5px solid #f5d76e; }
.rank-card.silver { background: linear-gradient(135deg,#f8fafc,#eef2ff); border: 1.5px solid #c7d2fe; }
.rank-card.bronze { background: linear-gradient(135deg,#fff7ed,#ffedd5); border: 1.5px solid #fdba74; }
.rank-card.neutral{ background: linear-gradient(135deg,#ffffff,#f8fafc); border: 1.5px solid #e2e8f0; }
.rank-card.danger { background: linear-gradient(135deg,#fef2f2,#fee2e2); border: 1.5px solid #fca5a5; }
.rank-medal { font-size: 1.8rem; line-height: 1; }
.rank-crop-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0.1rem 0 0 0;
    text-transform: capitalize;
}

/* ── Tags ── */
.tag {
    display: inline-block;
    padding: 0.22rem 0.7rem;
    border-radius: 999px;
    font-size: 0.71rem;
    font-weight: 700;
    margin: 0.2rem 0.25rem 0 0;
}
.tag-good { background: #dcfce7; color: #166534; }
.tag-warn { background: #fef3c7; color: #92400e; }
.tag-bad  { background: #fee2e2; color: #991b1b; }

/* ── Mini stat ── */
.mini-stat {
    background: #f0f9f1;
    border-radius: 12px;
    padding: 0.7rem 0.9rem;
    text-align: center;
    border: 1px solid #c8e6cc;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.mini-stat-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.08rem;
    font-weight: 700;
    color: #15803d;
}
.mini-stat-lbl {
    font-size: 0.65rem;
    color: #6b7c6d;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

/* ── Score bar ── */
.score-bar-wrap {
    background: #dde8de;
    border-radius: 4px; height: 5px;
    margin-top: 0.4rem; overflow: hidden;
}
.score-bar { height: 5px; border-radius: 4px; }
.score-bar.high { background: linear-gradient(90deg,#16a34a,#4ade80); }
.score-bar.mid  { background: linear-gradient(90deg,#d97706,#fbbf24); }
.score-bar.low  { background: linear-gradient(90deg,#dc2626,#f87171); }

/* ── Pipeline steps ── */
.pipeline-step {
    background: white;
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    border-left: 4px solid #22c55e;
    margin-bottom: 0.7rem;
    box-shadow: 0 2px 8px rgba(15,23,42,0.05);
}
.pipeline-step-num   { font-size:0.68rem; font-weight:700; color:#16a34a; text-transform:uppercase; letter-spacing:0.12em; }
.pipeline-step-title { font-family:'Space Grotesk',sans-serif; font-weight:700; color:#0f172a; margin:0.15rem 0 0.2rem; font-size:0.95rem; }
.pipeline-step-desc  { color:#4b6351; font-size:0.82rem; line-height:1.6; }

/* ── Expander fix — stop sidebar color bleeding ── */
[data-testid="stExpander"] {
    background: white !important;
    border-radius: 14px !important;
    border: 1px solid #d1fae5 !important;
}
[data-testid="stExpander"] summary {
    color: #0f172a !important;
    font-weight: 600 !important;
    background: white !important;
}
[data-testid="stExpander"] summary:hover {
    background: #f0fdf4 !important;
    color: #166534 !important;
}
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary div {
    color: #0f172a !important;
    background: transparent !important;
}
[data-testid="stExpander"] summary:hover span,
[data-testid="stExpander"] summary:hover p,
[data-testid="stExpander"] summary:hover div {
    color: #166534 !important;
    background: transparent !important;
}
[data-testid="stExpander"] details,
[data-testid="stExpander"] details > div,
[data-testid="stExpander"] details[open] {
    background: white !important;
    color: #0f172a !important;
}
[data-testid="stExpander"] details > div * {
    color: #1a2e1a !important;
    background: transparent !important;
}

/* ── Warn banner ── */
.warn-banner {
    background: linear-gradient(135deg,#fff7ed,#ffedd5);
    border: 1px solid #fdba74;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
}
.warn-banner-title { font-family:'Space Grotesk',sans-serif; font-weight:700; color:#9a3412; font-size:0.9rem; margin-bottom:0.2rem; text-transform:capitalize; }
.warn-banner-body  { color:#7c3b1e; font-size:0.82rem; }

/* ── Table ── */
thead tr th {
    background: #f0f9f1 !important;
    color: #166534 !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.09em !important;
    text-transform: uppercase !important;
    padding: 0.75rem 1rem !important;
    border-bottom: 2px solid #c8e6cc !important;
}
tbody tr td {
    color: #1a2e1a !important;
    font-size: 0.88rem !important;
    padding: 0.65rem 1rem !important;
    border-bottom: 1px solid #edf4ee !important;
    background: white !important;
}
tbody tr:hover td { background: #f0fdf4 !important; }
table {
    border-radius: 16px !important;
    overflow: hidden !important;
    border: 1px solid #d1fae5 !important;
    width: 100% !important;
    border-collapse: collapse !important;
}

/* ── Empty state ── */
.empty-state { text-align:center; padding:4rem 2rem; }
.empty-state-icon  { font-size:4rem; margin-bottom:1rem; }
.empty-state-title { font-family:'Space Grotesk',sans-serif; font-size:1.8rem; font-weight:700; color:#0f172a; margin-bottom:0.5rem; }
.empty-state-sub   { font-size:0.95rem; max-width:440px; margin:0 auto; color:#64748b; line-height:1.7; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#14532d 0%,#166534 100%);
    border-right: 1px solid rgba(255,255,255,0.07);
}
section[data-testid="stSidebar"] * { color: #ecfdf5 !important; }
section[data-testid="stSidebar"] .stSlider > label,
section[data-testid="stSidebar"] .stSelectbox > label {
    color: #86efac !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.10) !important; }
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg,#16a34a,#22c55e) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    height: 3rem !important;
    box-shadow: 0 8px 24px rgba(34,197,94,0.28) !important;
    transition: all 0.2s ease !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    box-shadow: 0 12px 32px rgba(34,197,94,0.38) !important;
    transform: translateY(-1px) !important;
}
.sidebar-logo    { font-family:'Space Grotesk',sans-serif; font-size:1.55rem; font-weight:700; color:white !important; margin-bottom:0.2rem; }
.sidebar-tagline { font-size:0.75rem; color:rgba(255,255,255,0.60) !important; margin-bottom:1.5rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def score_color(s):
    return "high" if s >= 70 else "mid" if s >= 35 else "low"

def card_class(rank, score):
    if score < 20:  return "danger"
    if rank == 0:   return "gold"
    if rank == 1:   return "silver"
    if rank == 2:   return "bronze"
    return "neutral"

def render_tags(tags):
    return "".join(
        f'<span class="tag tag-{kind}">{label}</span>'
        for label, kind in tags
    )

def score_bar(s):
    cls = score_color(s)
    return (f'<div class="score-bar-wrap">'
            f'<div class="score-bar {cls}" style="width:{min(s,100)}%"></div>'
            f'</div>')

def mini_stat(val, unit, label, color="#15803d"):
    return f"""
    <div class="mini-stat">
        <div class="mini-stat-val" style="color:{color}">
            {val}<span style="font-size:0.65rem;color:#6b7c6d;font-weight:400"> {unit}</span>
        </div>
        <div class="mini-stat-lbl">{label}</div>
    </div>"""


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🌾 CropRotate AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Precision Agriculture · KNN + Random Forest</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**🧪 SOIL NUTRIENTS**")
    N = st.slider("Nitrogen — N",    0, 140,  20)
    P = st.slider("Phosphorus — P",  5, 145,  50)
    K = st.slider("Potassium — K",   5, 205,  40)

    st.markdown("---")
    st.markdown("**🌡️ CLIMATE**")
    temperature = st.slider("Temperature (°C)", 10,  45, 25)
    humidity    = st.slider("Humidity (%)",     20, 100, 70)
    rainfall    = st.slider("Rainfall (mm)",    20, 300, 100)

    st.markdown("---")
    st.markdown("**⚗️ SOIL CHEMISTRY**")
    ph = st.slider("Soil pH", 3.0, 10.0, 6.5, step=0.1)

    st.markdown("---")
    st.markdown("**🌾 PREVIOUS CROP**")
    previous_crop = st.selectbox(
        "Previous crop",
        options=sorted(crop_info.keys()),
        label_visibility="collapsed"
    )

    if previous_crop in heavy_feeders:
        st.markdown(
            f'<div style="background:rgba(255,160,0,0.15);border-radius:10px;'
            f'padding:0.6rem 0.8rem;margin-top:0.5rem;font-size:0.8rem;color:#fbbf24;">'
            f'⚠️ <b>{previous_crop.title()}</b> is a heavy nitrogen consumer. '
            f'Legumes strongly recommended next.</div>',
            unsafe_allow_html=True
        )
    elif previous_crop in nitrogen_fixers:
        st.markdown(
            f'<div style="background:rgba(74,222,128,0.12);border-radius:10px;'
            f'padding:0.6rem 0.8rem;margin-top:0.5rem;font-size:0.8rem;color:#86efac;">'
            f'✅ <b>{previous_crop.title()}</b> fixed nitrogen. '
            f'Soil is well prepared.</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    run = st.button("🔍  Analyse & Recommend", use_container_width=True, type="primary")


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">AI · KNN + Random Forest Pipeline</div>
    <h1 class="hero-title">Precision <span>Crop Rotation</span> Advisor</h1>
    <p class="hero-sub">
        Input soil sensor readings and your previous crop.
        The two-stage ML pipeline finds environmentally compatible crops,
        then ranks them by agronomic soundness using FAO &amp; ICAR guidelines.
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HOW IT WORKS
# ─────────────────────────────────────────────
with st.expander("📖  How the Pipeline Works", expanded=False):
    ca, cb = st.columns(2)
    with ca:
        st.markdown("""
        <div class="pipeline-step">
            <div class="pipeline-step-num">Stage 1</div>
            <div class="pipeline-step-title">🔭 KNN Environmental Retrieval</div>
            <div class="pipeline-step-desc">
                Searches 2,200 historical field records using scaled Euclidean
                distance across 7 soil and climate features. Returns the 50 most
                similar historical fields and counts which crops survived there.
                Outputs a shortlist of <b>environmentally viable candidates</b>
                with a match percentage.
            </div>
        </div>""", unsafe_allow_html=True)
    with cb:
        st.markdown("""
        <div class="pipeline-step">
            <div class="pipeline-step-num">Stage 2</div>
            <div class="pipeline-step-title">🌲 Random Forest Agronomic Ranker</div>
            <div class="pipeline-step-desc">
                Trained on 484 crop-pair combinations labelled using
                FAO wheat-legume rotation guides and ICAR rice-based
                cropping system publications. Scores each candidate on
                rotation soundness, soil impact, and economics.
                <b>Final score = 70% RF confidence + 30% environmental match.</b>
            </div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
if run:
    with st.spinner("🌱 Analysing soil conditions and rotation logic..."):
        results = recommend_crop(
            previous_crop=previous_crop,
            N=N, P=P, K=K,
            temperature=temperature,
            humidity=humidity,
            ph=ph,
            rainfall=rainfall
        )

    top   = results[results['final_score'] >= 20].head(5)
    risky = results[results['final_score'] <  20]

    # ── Soil snapshot ──
    st.markdown('<div class="section-header">📡 Current Field Snapshot</div>',
                unsafe_allow_html=True)
    cols = st.columns(7)
    for col, (val, unit, label) in zip(cols, [
        (N, "N", "Nitrogen"), (P, "P", "Phosphorus"), (K, "K", "Potassium"),
        (temperature, "°C", "Temperature"), (humidity, "%", "Humidity"),
        (ph, "pH", "Soil pH"), (rainfall, "mm", "Rainfall")
    ]):
        col.markdown(mini_stat(val, unit, label), unsafe_allow_html=True)

    # ── Previous crop pill ──
    pill_color = "#c05e00" if previous_crop in heavy_feeders else \
                 "#166534" if previous_crop in nitrogen_fixers else "#334155"
    pill_label = "Heavy Feeder"   if previous_crop in heavy_feeders  else \
                 "Nitrogen Fixer" if previous_crop in nitrogen_fixers else "Standard Crop"
    st.markdown(
        f'<div style="margin:0.9rem 0 1.5rem;font-size:0.9rem;color:#334155;">'
        f'Previous crop: <b style="color:{pill_color};text-transform:capitalize">'
        f'{previous_crop}</b>&nbsp;&nbsp;'
        f'<span style="background:{pill_color}18;color:{pill_color};padding:3px 12px;'
        f'border-radius:999px;font-size:0.74rem;font-weight:700;">{pill_label}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Top recommendations ──
    st.markdown('<div class="section-header">🏆 Top Recommendations</div>',
                unsafe_allow_html=True)
    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]

    if top.empty:
        st.info("No strong candidates found for these conditions. Try adjusting the sliders.")
    else:
        for rank, (_, row) in enumerate(top.iterrows()):
            cc = card_class(rank, row['final_score'])
            sc = score_color(row['final_score'])
            sc_col = "#16a34a" if sc=="high" else "#d97706" if sc=="mid" else "#dc2626"

            left, right = st.columns([3, 2])
            with left:
                st.markdown(f"""
                <div class="rank-card {cc}">
                    <div style="display:flex;align-items:flex-start;gap:0.8rem;">
                        <div class="rank-medal">{medals[rank]}</div>
                        <div style="flex:1">
                            <div class="rank-crop-name">{row['crop']}</div>
                            <div style="margin-top:0.5rem">{render_tags(row['tags'])}</div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

            with right:
                s1, s2, s3 = st.columns(3)
                s1.markdown(
                    f'<div class="mini-stat">'
                    f'<div class="mini-stat-val" style="color:{sc_col}">{row["final_score"]}%</div>'
                    f'<div class="mini-stat-lbl">Final Score</div>'
                    f'{score_bar(row["final_score"])}</div>',
                    unsafe_allow_html=True
                )
                s2.markdown(
                    f'<div class="mini-stat">'
                    f'<div class="mini-stat-val" style="color:#1d4ed8">{row["ai_confidence"]}%</div>'
                    f'<div class="mini-stat-lbl">AI Confidence</div>'
                    f'{score_bar(row["ai_confidence"])}</div>',
                    unsafe_allow_html=True
                )
                s3.markdown(
                    f'<div class="mini-stat">'
                    f'<div class="mini-stat-val" style="color:#7c3aed">{row["env_match"]}%</div>'
                    f'<div class="mini-stat-lbl">Env Match</div>'
                    f'{score_bar(row["env_match"])}</div>',
                    unsafe_allow_html=True
                )

    # ── Crops to avoid ──
    if not risky.empty:
        st.markdown('<div class="section-header">🚫 Crops to Avoid This Season</div>',
                    unsafe_allow_html=True)
        warn_cols = st.columns(min(len(risky), 3))
        for i, (_, row) in enumerate(risky.iterrows()):
            reason = " · ".join(lbl for lbl, _ in row['tags']) if row['tags'] else "Sub-optimal rotation"
            warn_cols[i % 3].markdown(f"""
            <div class="warn-banner">
                <div class="warn-banner-title">⚠️ {row['crop'].title()}</div>
                <div class="warn-banner-body">Score: <b>{row['final_score']}%</b><br>{reason}</div>
            </div>""", unsafe_allow_html=True)

    # ── Full table ──
    st.markdown('<div class="section-header">📋 Complete Rankings</div>',
                unsafe_allow_html=True)

    display_df = results[[
        'crop', 'final_score', 'ai_confidence', 'env_match',
        'market_price', 'profitability', 'water_req'
    ]].copy()
    display_df.index = range(1, len(display_df) + 1)
    display_df.columns = [
        'Crop', 'Final Score %', 'AI Confidence %',
        'Env Match %', 'Market Price (USD/t)', 'Profitability /10', 'Water Need'
    ]
    display_df['Crop'] = display_df['Crop'].str.title()

    st.markdown("""
    <style>
    thead tr th {
        background: #f0f9f1 !important; color: #166534 !important;
        font-size: 0.7rem !important; font-weight: 700 !important;
        letter-spacing: 0.09em !important; text-transform: uppercase !important;
        padding: 0.75rem 1rem !important; border-bottom: 2px solid #bbf7d0 !important;
    }
    tbody tr td {
        color: #1a2e1a !important; font-size: 0.88rem !important;
        padding: 0.65rem 1rem !important; border-bottom: 1px solid #dcfce7 !important;
        background: white !important;
    }
    tbody tr:hover td { background: #f0fdf4 !important; }
    table {
        border-radius: 16px !important; overflow: hidden !important;
        border: 1px solid #bbf7d0 !important; width: 100% !important;
        border-collapse: collapse !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.table(display_df)

    st.caption(
        "Final Score = 70% RF agronomic confidence + 30% KNN environmental match  ·  "
        "Trained on FAO & ICAR rotation guidelines  ·  5-Fold CV F1: 0.920 ± 0.022"
    )

# ─────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────
else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">🌱</div>
        <div class="empty-state-title">Ready to Analyse Your Field</div>
        <div class="empty-state-sub">
            Set your soil readings and previous crop in the sidebar,
            then click <b>Analyse &amp; Recommend</b>.
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">📖 How It Works</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, (icon, title, desc) in zip([c1, c2, c3], [
        ("🧪", "Input Soil Data",
         "Enter NPK levels, temperature, humidity, pH and rainfall from your field sensors."),
        ("🔭", "KNN Retrieval",
         "Finds the 50 most similar historical fields from 2,200 records and surfaces viable crops."),
        ("🌲", "RF Ranking",
         "Scores each candidate using agronomic rotation rules from FAO & ICAR guidelines.")
    ]):
        col.markdown(f"""
        <div class="pipeline-step" style="text-align:center;border-left:none;
             border-top:4px solid #22c55e;padding-top:1.2rem;">
            <div style="font-size:2rem;margin-bottom:0.5rem">{icon}</div>
            <div class="pipeline-step-title">{title}</div>
            <div class="pipeline-step-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)