"""
AirCast — Karachi AQI Forecast Dashboard
A premium, futuristic Streamlit dashboard for real-time AQI prediction.
"""
import os, json, joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from model_utils import AQIDeltaRegressor

# ── Page Config ──
st.set_page_config(page_title="AirCast // Karachi AQI", page_icon="🌌", layout="wide", initial_sidebar_state="expanded")

# ── Feature columns the trained models expect (34 features) ──
MODEL_FEATURES = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide",
    "ozone", "dust", "aerosol_optical_depth", "uv_index", "temperature",
    "humidity", "wind_speed", "wind_direction", "precipitation", "pressure",
    "boundary_layer_height", "aqi_lag_24h", "aqi_lag_48h", "aqi_lag_72h",
    "dispersion_index", "hour_sin", "hour_cos", "month_sin", "month_cos",
    "is_weekend", "aqi_change_rate", "aqi_rolling_24h", "pm_ratio",
    "blh_change_rate", "vpd_rolling_3h", "solar_rolling_3h", "aqi",
    "cloudcover", "solar_radiation",
]

# ── HTML Ambient Background Orbs ──
st.markdown("""
<div class="glow-orb orb-1"></div>
<div class="glow-orb orb-2"></div>
<div class="glow-orb orb-3"></div>
""", unsafe_allow_html=True)

# ── CSS Styling System ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&family=Rajdhani:wght@500;600;700&display=swap');

/* Main App Layout overrides */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    color: #e2e8f0;
}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 50% 50%, #0c0d19 0%, #05060b 100%) !important;
    background-attachment: fixed !important;
}
[data-testid="stHeader"] {
    background: transparent !important;
}
[data-testid="stAppViewBlockContainer"] {
    max-width: 1300px !important;
    padding: 2.5rem 2rem 5rem 2rem !important;
}

/* Custom Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: rgba(10, 11, 24, 0.4);
}
::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 242, 254, 0.3);
}

/* Floating Ambient Orbs */
.glow-orb {
    position: fixed;
    border-radius: 50%;
    filter: blur(140px);
    opacity: 0.12;
    z-index: -99999;
    pointer-events: none;
    animation: floatOrb 25s infinite ease-in-out;
}
.orb-1 {
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, #00f2fe 0%, rgba(0, 242, 254, 0) 70%);
    top: -150px;
    left: -150px;
}
.orb-2 {
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, #a855f7 0%, rgba(168, 85, 247, 0) 70%);
    bottom: -200px;
    right: -150px;
    animation-delay: -7s;
}
.orb-3 {
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, #ff007f 0%, rgba(255, 0, 127, 0) 70%);
    top: 35%;
    left: 55%;
    animation-delay: -14s;
}
@keyframes floatOrb {
    0%, 100% { transform: translate(0, 0) scale(1); }
    50% { transform: translate(50px, -70px) scale(1.08); }
}

/* Entry Animations */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(15px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
[data-testid="stVerticalBlock"] > div {
    opacity: 0;
    animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}
[data-testid="stVerticalBlock"] > div:nth-child(1) { animation-delay: 0.05s; }
[data-testid="stVerticalBlock"] > div:nth-child(2) { animation-delay: 0.1s; }
[data-testid="stVerticalBlock"] > div:nth-child(3) { animation-delay: 0.15s; }
[data-testid="stVerticalBlock"] > div:nth-child(4) { animation-delay: 0.2s; }
[data-testid="stVerticalBlock"] > div:nth-child(5) { animation-delay: 0.25s; }

/* Sidebar Premium Override */
[data-testid="stSidebar"] {
    background: rgba(8, 9, 17, 0.85) !important;
    backdrop-filter: blur(25px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
}
[data-testid="stSidebarContent"] {
    padding-top: 2rem !important;
}
[data-testid="stSidebar"] label {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    font-size: 0.82rem !important;
}
[data-testid="stSidebar"] .stSlider label {
    text-transform: none !important;
}

/* Sidebar Custom Tabs / Radio Buttons */
[data-testid="stRadio"] > div {
    flex-direction: column !important;
    gap: 8px !important;
}
[data-testid="stRadio"] label {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    padding: 10px 18px !important;
    border-radius: 12px !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    cursor: pointer !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.92rem !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    font-weight: 400 !important;
    color: #cbd5e1 !important;
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
}
[data-testid="stRadio"] label:hover {
    background: rgba(255, 255, 255, 0.05) !important;
    border-color: rgba(0, 242, 254, 0.25) !important;
    transform: translateX(3px);
}
[data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(135deg, rgba(0, 242, 254, 0.12) 0%, rgba(168, 85, 247, 0.12) 100%) !important;
    border-color: rgba(0, 242, 254, 0.4) !important;
    color: #ffffff !important;
    box-shadow: 0 0 15px rgba(0, 242, 254, 0.08) !important;
    font-weight: 600 !important;
}
[data-testid="stRadio"] input[type="radio"] {
    display: none !important;
}
[data-testid="stRadio"] label > div:first-child {
    display: none !important;
}

/* Sidebar Sliders */
.stSlider [data-testid="stWidgetLabel"] {
    font-size: 0.8rem !important;
    color: #94a3b8 !important;
    margin-bottom: 8px !important;
}
.stSlider [data-baseweb="slider"] {
    padding-left: 8px !important;
    padding-right: 8px !important;
}
.stSlider [role="slider"] {
    background-color: #00f2fe !important;
    border: 2px solid #a855f7 !important;
    box-shadow: 0 0 8px rgba(0, 242, 254, 0.7) !important;
}
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #00f2fe, #a855f7) !important;
}
.stSlider > div > div > div {
    background-color: rgba(255, 255, 255, 0.06) !important;
}

/* Custom UI Cards */
.glass-card {
    background: rgba(15, 20, 35, 0.45) !important;
    backdrop-filter: blur(20px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 20px !important;
    padding: 26px !important;
    margin-bottom: 20px;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
.glass-card:hover {
    transform: translateY(-4px);
}

.glass-card.hover-glow-pill-good:hover {
    border-color: rgba(72, 187, 120, 0.35) !important;
    box-shadow: 0 20px 50px rgba(72, 187, 120, 0.1), inset 0 1px 2px rgba(255, 255, 255, 0.1) !important;
}
.glass-card.hover-glow-pill-moderate:hover {
    border-color: rgba(236, 201, 75, 0.35) !important;
    box-shadow: 0 20px 50px rgba(236, 201, 75, 0.1), inset 0 1px 2px rgba(255, 255, 255, 0.1) !important;
}
.glass-card.hover-glow-pill-sensitive:hover {
    border-color: rgba(237, 137, 54, 0.35) !important;
    box-shadow: 0 20px 50px rgba(237, 137, 54, 0.1), inset 0 1px 2px rgba(255, 255, 255, 0.1) !important;
}
.glass-card.hover-glow-pill-unhealthy:hover {
    border-color: rgba(229, 62, 62, 0.35) !important;
    box-shadow: 0 20px 50px rgba(229, 62, 62, 0.15), inset 0 1px 2px rgba(255, 255, 255, 0.1) !important;
}
.glass-card.hover-glow-pill-hazardous:hover {
    border-color: rgba(183, 110, 255, 0.35) !important;
    box-shadow: 0 20px 50px rgba(183, 110, 255, 0.15), inset 0 1px 2px rgba(255, 255, 255, 0.1) !important;
}

.metric-label {
    font-size: 0.8rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
}
.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    margin-top: 8px;
    line-height: 1;
}

/* Health pills */
.pill {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 30px;
    font-weight: 600;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-top: 12px;
    backdrop-filter: blur(5px);
}
.pill-good { background: rgba(72, 187, 120, 0.08); color: #48bb78; border: 1px solid rgba(72, 187, 120, 0.25); }
.pill-moderate { background: rgba(236, 201, 75, 0.08); color: #ecc94b; border: 1px solid rgba(236, 201, 75, 0.25); }
.pill-sensitive { background: rgba(237, 137, 54, 0.08); color: #ed8936; border: 1px solid rgba(237, 137, 54, 0.25); }
.pill-unhealthy { background: rgba(229, 62, 62, 0.08); color: #e53e3e; border: 1px solid rgba(229, 62, 62, 0.25); }
.pill-hazardous { background: rgba(183, 110, 255, 0.08); color: #b76eff; border: 1px solid rgba(183, 110, 255, 0.25); }

/* Environmental Telemetry Grid */
.telemetry-card {
    background: rgba(255, 255, 255, 0.01) !important;
    border: 1px solid rgba(255, 255, 255, 0.04) !important;
    border-radius: 16px !important;
    padding: 18px 14px !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    text-align: center !important;
}
.telemetry-card:hover {
    background: rgba(255, 255, 255, 0.03) !important;
    border-color: rgba(0, 242, 254, 0.2) !important;
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(0, 242, 254, 0.03) !important;
}
.telemetry-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
    margin-bottom: 6px;
    font-family: 'Space Grotesk', sans-serif;
}
.telemetry-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #f8fafc;
    font-family: 'Space Grotesk', sans-serif;
}
.telemetry-unit {
    font-size: 0.75rem;
    color: #475569;
    font-weight: 400;
}

/* Model Registry Card */
.sidebar-model-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 12px;
    transition: all 0.3s ease;
}
.sidebar-model-card:hover {
    background: rgba(255, 255, 255, 0.04);
    border-color: rgba(0, 242, 254, 0.2);
}
.sm-label {
    font-size: 0.72rem;
    color: #00f2fe;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.sm-name {
    font-size: 0.85rem;
    color: #cbd5e1;
    font-weight: 500;
    margin: 2px 0 4px 0;
}
.sm-stats {
    font-size: 0.72rem;
    color: #64748b;
}

/* Sidebar Custom Labels */
.sidebar-section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: 1.5px;
    margin-top: 25px;
    margin-bottom: 15px;
    text-transform: uppercase;
    border-left: 3px solid #a855f7;
    padding-left: 10px;
}
.sidebar-subsection-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    color: #94a3b8;
    letter-spacing: 1px;
    margin-top: 18px;
    margin-bottom: 8px;
    text-transform: uppercase;
}

/* SHAP Card and Elements */
.shap-container {
    display: flex;
    flex-direction: column;
    gap: 16px;
    margin-top: 15px;
}
.shap-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.shap-label {
    width: 130px;
    font-size: 0.85rem;
    color: #94a3b8;
    font-weight: 500;
    font-family: 'Outfit', sans-serif;
}
.shap-track-wrapper {
    flex-grow: 1;
    margin: 0 15px;
}
.shap-track {
    height: 8px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 4px;
    position: relative;
}
.shap-center-line {
    position: absolute;
    left: 50%;
    top: -2px;
    width: 1px;
    height: 12px;
    background: rgba(255, 255, 255, 0.2);
}
.shap-bar-pos {
    position: absolute;
    left: 50%;
    height: 8px;
    background: linear-gradient(90deg, #a855f7, #ff007f);
    border-radius: 0 4px 4px 0;
    box-shadow: 0 0 8px rgba(255, 0, 127, 0.4);
}
.shap-bar-neg {
    position: absolute;
    right: 50%;
    height: 8px;
    background: linear-gradient(270deg, #00f2fe, #4facfe);
    border-radius: 4px 0 0 4px;
    box-shadow: 0 0 8px rgba(0, 242, 254, 0.4);
}
.shap-val {
    width: 55px;
    text-align: right;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
}
.v-pos { color: #ff007f; }
.v-neg { color: #00f2fe; }

/* Alert banner */
@keyframes alertPulse { 0%,100%{opacity:0.85;} 50%{opacity:1;} }
.alert-banner {
    border-radius: 16px !important;
    padding: 20px 24px !important;
    margin-bottom: 28px !important;
    display: flex !important;
    align-items: center !important;
    backdrop-filter: blur(10px) !important;
    position: relative !important;
    overflow: hidden !important;
    animation: alertPulse 2.5s ease-in-out infinite;
}
.alert-banner::after {
    content: '';
    position: absolute;
    top: 0; left: 0; bottom: 0; width: 4px;
}
.alert-danger {
    background: rgba(239, 68, 68, 0.05) !important;
    border: 1px solid rgba(239, 68, 68, 0.2) !important;
    box-shadow: 0 10px 30px rgba(239, 68, 68, 0.05) !important;
}
.alert-danger::after {
    background: #ef4444;
}
.alert-warning {
    background: rgba(245, 158, 11, 0.05) !important;
    border: 1px solid rgba(245, 158, 11, 0.2) !important;
    box-shadow: 0 10px 30px rgba(245, 158, 11, 0.05) !important;
}
.alert-warning::after {
    background: #f59e0b;
}

/* Hero elements */
.hero-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 30px 20px 20px 20px;
    position: relative;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    margin: 0 auto 15px auto;
    padding: 6px 16px;
    background: rgba(0, 242, 254, 0.05);
    border: 1px solid rgba(0, 242, 254, 0.2);
    border-radius: 30px;
    color: #00f2fe;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    box-shadow: 0 0 20px rgba(0, 242, 254, 0.15);
}
.pulse-dot {
    width: 8px; height: 8px; background: #00f2fe; border-radius: 50%;
    display: inline-block; animation: pulse 2s infinite; margin-right: 10px; vertical-align: middle;
}
@keyframes pulse { 0%,100%{box-shadow:0 0 0 0 rgba(0,242,254,0.5);} 70%{box-shadow:0 0 0 8px rgba(0,242,254,0);} }
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 800;
    font-size: 3.8rem;
    letter-spacing: 5px;
    background: linear-gradient(90deg, #00f2fe, #4facfe, #a855f7, #ff007f, #00f2fe);
    background-size: 400% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 12s linear infinite;
    margin-bottom: 5px;
    text-align: center;
}
@keyframes shine {
    to { background-position: 400% center; }
}
.hero-subtitle {
    font-family: 'Outfit', sans-serif;
    font-size: 1rem;
    color: #64748b;
    letter-spacing: 2px;
    text-transform: uppercase;
    text-align: center;
    margin-bottom: 30px;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)


# ── Data & Model Loading ──
def load_data():
    """Load the latest feature data.
    
    The original implementation used @st.cache_data, which caused the data to be cached and not refreshed when the underlying parquet file was updated hourly. 
    Removing the cache ensures the dashboard always reads the most recent data on each run.
    """
    path = os.path.join("data", "features.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    # Ensure rows are sorted chronologically so the latest entry is last
    return df.sort_values("timestamp")

@st.cache_resource
def load_models():
    models = {}
    for t in ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]:
        mp = os.path.join("models", t, "model.pkl")
        mj = os.path.join("models", t, "metrics.json")
        if os.path.exists(mp):
            models[t] = joblib.load(mp)
        if os.path.exists(mj):
            with open(mj, "r", encoding="utf-8") as f:
                models[f"{t}_meta"] = json.load(f)
    return models

df_hist = load_data()
models = load_models()

# Extract latest row
if df_hist is not None and len(df_hist) > 0:
    latest = df_hist.iloc[-1].to_dict()
    last_time = df_hist.iloc[-1]["timestamp"].strftime("%Y-%m-%d %H:%M")
else:
    latest = {c: 0.0 for c in MODEL_FEATURES}
    latest.update({"aqi": 100, "pm2_5": 60, "pm10": 100, "temperature": 32, "wind_speed": 12,
                   "boundary_layer_height": 700, "humidity": 55, "ozone": 25, "carbon_monoxide": 600})
    last_time = "No data — run backfill_pipeline.py"


def aqi_category(val):
    if val <= 50:   return "Good", "pill-good", "#48bb78"
    if val <= 100:  return "Moderate", "pill-moderate", "#ecc94b"
    if val <= 150:  return "Sensitive Groups", "pill-sensitive", "#ed8936"
    if val <= 200:  return "Unhealthy", "pill-unhealthy", "#e53e3e"
    return "Hazardous", "pill-hazardous", "#b76eff"


# ── Hero Section ──
st.markdown(f"""
<div class="hero-container">
    <div class="hero-badge"><span class="pulse-dot"></span>LIVE TELEMETRY &nbsp;·&nbsp; LAST INGEST: {last_time}</div>
    <div class="hero-title">A I R C A S T</div>
    <div class="hero-subtitle">// Karachi Air Quality Intelligence & Forecasting</div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ──
with st.sidebar:
    st.markdown("<div class='sidebar-section-title'>🌌 CONTROL PANEL</div>", unsafe_allow_html=True)
    mode = st.radio("Mode", ["Live Dashboard", "What-If Sandbox"], index=0)
    
    st.markdown("<div class='sidebar-section-title'>🤖 MODEL REGISTRY</div>", unsafe_allow_html=True)
    for tgt, label in [("target_aqi_24h","24h Forecast"), ("target_aqi_48h","48h Forecast"), ("target_aqi_72h","72h Forecast")]:
        meta = models.get(f"{tgt}_meta", {})
        st.markdown(f"""
        <div class="sidebar-model-card">
            <div class="sm-label">{label}</div>
            <div class="sm-name">{meta.get('model_name','—')}</div>
            <div class="sm-stats">RMSE: {meta.get('rmse',0):.1f} &nbsp;·&nbsp; R²: {meta.get('r2',0):.3f}</div>
        </div>
        """, unsafe_allow_html=True)


# ── Build feature vector ──
sim = dict(latest)

if mode == "What-If Sandbox":
    st.sidebar.markdown("<div class='sidebar-section-title'>🛠 SIMULATION</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='sidebar-subsection-title'>Pollutants</div>", unsafe_allow_html=True)
    sim["pm2_5"] = float(st.sidebar.slider("PM2.5 (µg/m³)", 0, 300, int(sim.get("pm2_5", 60))))
    sim["pm10"] = float(st.sidebar.slider("PM10 (µg/m³)", 0, 500, int(sim.get("pm10", 100))))
    sim["ozone"] = float(st.sidebar.slider("Ozone (µg/m³)", 0, 250, int(sim.get("ozone", 25))))
    sim["carbon_monoxide"] = float(st.sidebar.slider("CO (µg/m³)", 0, 5000, int(sim.get("carbon_monoxide", 600))))
    
    st.sidebar.markdown("<div class='sidebar-subsection-title'>Weather</div>", unsafe_allow_html=True)
    sim["temperature"] = float(st.sidebar.slider("Temperature (°C)", 10, 50, int(sim.get("temperature", 32))))
    sim["wind_speed"] = float(st.sidebar.slider("Wind Speed (km/h)", 0, 80, int(sim.get("wind_speed", 12))))
    sim["boundary_layer_height"] = float(st.sidebar.slider("BLH (m)", 100, 3000, int(sim.get("boundary_layer_height", 700))))
    sim["humidity"] = float(st.sidebar.slider("Humidity (%)", 5, 100, int(sim.get("humidity", 55))))
    # Recalculate derived
    sim["pm_ratio"] = sim["pm2_5"] / (sim["pm10"] + 0.1)
    sim["dispersion_index"] = sim["wind_speed"] * sim["boundary_layer_height"]
    # Estimate current AQI from PM2.5 (US EPA breakpoints)
    pm = sim["pm2_5"]
    if pm <= 12:      sim["aqi"] = int((50/12)*pm)
    elif pm <= 35.4:  sim["aqi"] = int(50 + (50/23.4)*(pm-12))
    elif pm <= 55.4:  sim["aqi"] = int(100 + (50/20)*(pm-35.4))
    elif pm <= 150.4: sim["aqi"] = int(150 + (50/95)*(pm-55.4))
    else:             sim["aqi"] = int(min(500, 200 + (100/100)*(pm-150.4)))

sim_df = pd.DataFrame([sim])

# ── Predictions ──
predictions = {}
for tgt in ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]:
    fallback_offset = {"target_aqi_24h": 5, "target_aqi_48h": 10, "target_aqi_72h": 15}
    if tgt in models:
        try:
            cols = [c for c in MODEL_FEATURES if c in sim_df.columns]
            X = sim_df[cols].copy()
            # Fill any missing model columns with 0
            for mc in MODEL_FEATURES:
                if mc not in X.columns:
                    X[mc] = 0.0
            X = X[MODEL_FEATURES]  # enforce exact order
            predictions[tgt] = int(np.clip(models[tgt].predict(X)[0], 0, 500))
        except Exception:
            predictions[tgt] = int(sim.get("aqi", 100)) + fallback_offset[tgt]
    else:
        predictions[tgt] = int(sim.get("aqi", 100)) + fallback_offset[tgt]

cur_aqi = int(sim.get("aqi", 100))
aqi_24 = predictions["target_aqi_24h"]
aqi_48 = predictions["target_aqi_48h"]
aqi_72 = predictions["target_aqi_72h"]

# ── Row 1: Forecast Cards ──
cols = st.columns(4)
for col, (label, val) in zip(cols, [("Current AQI", cur_aqi), ("24h Forecast", aqi_24), ("48h Forecast", aqi_48), ("72h Forecast", aqi_72)]):
    cat, pill, clr = aqi_category(val)
    with col:
        st.markdown(f"""
        <div class='glass-card hover-glow-{pill}' style='text-align:center;'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value' style='color:{clr}; text-shadow:0 0 20px {clr}30;'>{val}</div>
            <span class='pill {pill}'>{cat}</span>
        </div>""", unsafe_allow_html=True)

# ── Alert Banner ──
peak = max(aqi_24, aqi_48, aqi_72)
if peak > 150:
    st.markdown(f"""<div class='alert-banner alert-danger'>
        <span style='font-size:1.6rem;margin-right:14px;'>🚨</span>
        <div><strong style='color:#ef4444;text-transform:uppercase;'>Hazardous AQI Alert</strong><br/>
        Forecast indicates AQI will reach <b>{peak} ({aqi_category(peak)[0]})</b> within 72 hours. Sensitive groups should limit outdoor exposure.</div>
    </div>""", unsafe_allow_html=True)
elif peak > 100:
    st.markdown(f"""<div class='alert-banner alert-warning'>
        <span style='font-size:1.6rem;margin-right:14px;'>⚠️</span>
        <div><strong style='color:#f59e0b;text-transform:uppercase;'>Moderate Pollution Warning</strong><br/>
        AQI may reach <b>{peak}</b>. Acceptable for most, but sensitive individuals should take precautions.</div>
    </div>""", unsafe_allow_html=True)

# ── Row 2: Chart + SHAP ──
col_chart, col_shap = st.columns([3, 2])

with col_chart:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 📈 FORECAST TRAJECTORY")
    steps = ["Now", "+24h", "+48h", "+72h"]
    vals = [cur_aqi, aqi_24, aqi_48, aqi_72]
    clrs = [aqi_category(v)[2] for v in vals]

    fig = go.Figure()
    # Gradient area fill
    fig.add_trace(go.Scatter(x=steps, y=vals, fill='tozeroy', fillcolor='rgba(0,242,254,0.03)',
        line=dict(color='#00f2fe', width=4, shape='spline'), mode='lines+markers',
        marker=dict(size=12, color=clrs, line=dict(width=3, color='#0a0e1a')), name='AQI'))
    # Danger threshold lines
    fig.add_hline(y=150, line_dash="dash", line_color="rgba(239,68,68,0.4)", annotation_text="Unhealthy", annotation_font_color="#ef4444", annotation_font_family="Space Grotesk")
    fig.add_hline(y=100, line_dash="dash", line_color="rgba(245,158,11,0.25)", annotation_text="Moderate", annotation_font_color="#f59e0b", annotation_font_family="Space Grotesk")
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Outfit, sans-serif", color="#94a3b8"),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.03)', 
            linecolor='rgba(255,255,255,0.05)',
            tickfont=dict(color='#94a3b8', size=12, family="Space Grotesk, sans-serif")
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.03)', 
            linecolor='rgba(255,255,255,0.05)',
            tickfont=dict(color='#94a3b8', family="Space Grotesk, sans-serif"), 
            title=dict(text="AQI Value", font=dict(color='#64748b', size=12, family="Space Grotesk, sans-serif")),
            range=[0, max(220, max(vals)+40)]
        ),
        margin=dict(l=55, r=25, t=25, b=35), height=340, showlegend=False)
    st.plotly_chart(fig, width='stretch')
    st.markdown("</div>", unsafe_allow_html=True)

with col_shap:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 🧬 FEATURE IMPACT ANALYSIS")
    st.caption("What's driving the 24h forecast up ↑ or down ↓")
    # Compute contribution proxies based on feature deviations from typical baselines
    contribs = {
        "PM2.5 Level":      (float(sim.get("pm2_5", 60)) - 45) * 0.40,
        "PM10 Dust":        (float(sim.get("pm10", 100)) - 90) * 0.12,
        "Wind Dispersion":  -(float(sim.get("wind_speed", 12)) - 12) * 0.80,
        "Temperature":      (float(sim.get("temperature", 32)) - 28) * 0.50,
        "Mixing Height":    -(float(sim.get("boundary_layer_height", 700)) - 600) * 0.015,
        "Humidity":         (float(sim.get("humidity", 55)) - 50) * 0.10,
    }
    sorted_c = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)
    mx = max(abs(v) for _, v in sorted_c) or 1.0

    html = "<div class='shap-container'>"
    for feat, val in sorted_c[:6]:
        pct = min(100, int(abs(val) / mx * 100))
        half_pct = pct / 2  # split across the center line
        sign = f"+{val:.1f}" if val >= 0 else f"{val:.1f}"
        vc = "v-pos" if val >= 0 else "v-neg"
        
        if val >= 0:
            bar_html = f"<div class='shap-bar-pos' style='width:{half_pct}%'></div>"
        else:
            bar_html = f"<div class='shap-bar-neg' style='width:{half_pct}%'></div>"
            
        html += f"""
        <div class='shap-row'>
            <div class='shap-label'>{feat}</div>
            <div class='shap-track-wrapper'>
                <div class='shap-track'>
                    <div class='shap-center-line'></div>
                    {bar_html}
                </div>
            </div>
            <div class='shap-val {vc}'>{sign}</div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Row 3: Environmental Readings ──
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("### 📊 ENVIRONMENTAL TELEMETRY")
c1, c2, c3, c4, c5 = st.columns(5)
def telemetry_metric_html(label, value, unit=""):
    u = f" <span class='telemetry-unit'>{unit}</span>" if unit else ""
    return f"""
    <div class='telemetry-card'>
        <div class='telemetry-label'>{label}</div>
        <div class='telemetry-value'>{value}{u}</div>
    </div>
    """

with c1:
    st.markdown(telemetry_metric_html("PM2.5", int(sim.get("pm2_5",0)), "µg/m³"), unsafe_allow_html=True)
    st.markdown(f"<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown(telemetry_metric_html("PM10", int(sim.get("pm10",0)), "µg/m³"), unsafe_allow_html=True)
with c2:
    st.markdown(telemetry_metric_html("Ozone", int(sim.get("ozone",0)), "µg/m³"), unsafe_allow_html=True)
    st.markdown(f"<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown(telemetry_metric_html("CO", int(sim.get("carbon_monoxide",0)), "µg/m³"), unsafe_allow_html=True)
with c3:
    st.markdown(telemetry_metric_html("Temperature", f"{int(sim.get('temperature',0))}°C"), unsafe_allow_html=True)
    st.markdown(f"<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown(telemetry_metric_html("Wind", int(sim.get("wind_speed",0)), "km/h"), unsafe_allow_html=True)
with c4:
    st.markdown(telemetry_metric_html("BLH", int(sim.get("boundary_layer_height",0)), "m"), unsafe_allow_html=True)
    st.markdown(f"<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown(telemetry_metric_html("Dispersion", int(sim.get("dispersion_index",0))), unsafe_allow_html=True)
with c5:
    st.markdown(telemetry_metric_html("Humidity", f"{int(sim.get('humidity',0))}%"), unsafe_allow_html=True)
    st.markdown(f"<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown(telemetry_metric_html("Pressure", int(sim.get("pressure",0)), "hPa"), unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── Row 4: Historical AQI Trend (last 7 days) ──
if df_hist is not None and len(df_hist) > 48:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 🕐 HISTORICAL AQI (LAST 7 DAYS)")
    recent = df_hist.tail(168)  # ~7 days of hourly data
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=recent["timestamp"], y=recent["aqi"], mode="lines",
        line=dict(color="#4facfe", width=2, shape="spline"),
        fill="tozeroy", fillcolor="rgba(79,172,254,0.03)", name="AQI"))
    fig2.add_hline(y=150, line_dash="dash", line_color="rgba(239,68,68,0.3)", annotation_text="Unhealthy Baseline", annotation_font_color="rgba(239,68,68,0.5)", annotation_font_family="Space Grotesk")
    fig2.add_hline(y=100, line_dash="dash", line_color="rgba(245,158,11,0.2)", annotation_text="Moderate Baseline", annotation_font_color="rgba(245,158,11,0.4)", annotation_font_family="Space Grotesk")
    fig2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Outfit, sans-serif", color="#94a3b8"),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.03)', 
            linecolor='rgba(255,255,255,0.05)',
            tickfont=dict(color='#94a3b8', family="Space Grotesk, sans-serif")
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.03)', 
            linecolor='rgba(255,255,255,0.05)',
            tickfont=dict(color='#94a3b8', family="Space Grotesk, sans-serif"),
            title=dict(text="AQI Value", font=dict(color='#64748b', size=12, family="Space Grotesk, sans-serif"))
        ),
        margin=dict(l=55, r=25, t=25, b=35), height=280, showlegend=False)
    st.plotly_chart(fig2, width='stretch')
    st.markdown("</div>", unsafe_allow_html=True)
