# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import requests
from io import BytesIO
import re

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# ───── PAGE CONFIG & GLOBAL CSS ─────
st.set_page_config(page_title="Restaurant Finder", page_icon="🍽️", layout="wide")

FOOD_BG = (
    "https://images.unsplash.com/photo-1552566626-52f8b828add9"
    "?auto=format&fit=crop&w=1350&q=80"
)

st.markdown(f"""
<style>
/* Full-page food background */
html, body, .stApp {{
  background: url('{FOOD_BG}') no-repeat center center fixed;
  background-size: cover;
}}

/* Overlay to dim the bg for readability */
.stApp::before {{
  content: "";
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(255,255,255,0.85);
  z-index: -1;
}}

/* Sidebar “filter box” */
[data-testid="stSidebar"] > div:first-child {{
  background: #fff;
  padding: 20px;
  border-radius: 12px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.1);
  margin: 16px;
}}

/* Filter labels */
.stSidebar label {{
  font-weight: 600;
  margin-top: 12px;
}}

/* Badges for keywords */
.badge {{
  display:inline-block;
  background:#ffb300;
  color:#fff;
  border-radius:4px;
  padding:4px 8px;
  margin:4px 4px 4px 0;
  font-size:0.85rem;
}}

/* Review snippet cards */
.snippet-card {{
  background:#fff;
  border-left:4px solid #43a047;
  padding:12px 16px;
  margin:8px 0;
  border-radius:6px;
  box-shadow:0 1px 4px rgba(0,0,0,0.1);
}}

/* Formatted address */
.address {{
  font-style:italic;
  color:#555;
  margin-bottom:12px;
}}
</style>
""", unsafe_allow_html=True)

# ───── DATA LOAD ─────
DATA_URL = "https://drive.google.com/uc?export=download&id=1iRWeGaDybybQ2eiTCyEgyXDYbH5FpFup"

@st.cache_data
def load_data(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_csv(BytesIO(r.content))

df_raw = load_data(DATA_URL)

# ───── MODEL TRAINING ─────
@st.cache_resource
def train_model(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["google_rating","price","popularity","sentiment"])
    num_feats = ["price","popularity","sentiment"]
    cat_feats = ["category"]
    X, y = df[num_feats + cat_feats], df["google_rating"]

    prep = ColumnTransformer([
        ("num", StandardScaler(), num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
    ])

    models = {
        "RF": RandomForestRegressor(n_estimators=150, random_state=42),
        "GB": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
    }

    best_rmse, best_pipe = np.inf, None
    for m in models.values():
        pipe = Pipeline([("prep", prep), ("reg", m)])
        pipe.fit(X, y)
        rmse = mean_squared_error(y, pipe.predict(X)) ** 0.5
        if rmse < best_rmse:
            best_rmse, best_pipe = rmse, pipe

    best_pipe.fit(X, y)
    df["predicted_rating"] = best_pipe.predict(X).round(1)
    return df

df = train_model(df_raw)

# ───── FILTERS ─────
st.sidebar.header("Filters")

# City
if "city" in df.columns:
    cities = ["All"] + sorted(df["city"].dropna().unique())
    city = st.sidebar.selectbox("City", cities)
    if city != "All":
