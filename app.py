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
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# ───── PAGE CONFIG & GLOBAL CSS ─────
st.set_page_config(page_title="🍽️ Restaurant Finder", layout="wide")

FOOD_BG = "https://images.unsplash.com/photo-1552566626-52f8b828add9?auto=format&fit=crop&w=1350&q=80"

st.markdown(f"""
<style>
html, body, .stApp {{
  background: url('{FOOD_BG}') no-repeat center center fixed;
  background-size: cover;
}}
.stApp::before {{
  content: "";
  position: fixed; top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(255,255,255,0.4); z-index: -1;
}}
.block-container {{
  background: rgba(255,255,255,0.95); border-radius:12px;
  padding:2rem !important; box-shadow:0 4px 20px rgba(0,0,0,0.1);
}}
/* Sidebar banner */
.sidebar-banner {{
  text-align:center; margin-bottom:16px;
}}
.sidebar-banner img {{
  width:80px; border-radius:50%;
}}
/* Expander cards */
.stExpander > div {{
  background:#fff; border-radius:8px; padding:12px;
  margin-bottom:12px; box-shadow:0 2px 8px rgba(0,0,0,0.05);
}}
/* Expander summary icons */
.stExpanderSummary {{
  font-size:1rem !important; font-weight:600;
}}
/* Reset button */
.reset-btn {{
  width:100%; margin-top:16px; background:#43a047; color:#fff;
  border:none; padding:10px; border-radius:8px; cursor:pointer;
}}
</style>
""", unsafe_allow_html=True)

# ───── DATA LOAD ─────
DATA_URL = "https://drive.google.com/uc?export=download&id=1iRWeGaDybybQ2eiTCyEgyXDYbH5FpFup"
@st.cache_data
def load_data(url):
    r = requests.get(url, timeout=60); r.raise_for_status()
    return pd.read_csv(BytesIO(r.content))
df_raw = load_data(DATA_URL)

# ───── MODEL TRAINING ─────
@st.cache_resource
def train_model(df):
    df = df.dropna(subset=["google_rating","price","popularity","sentiment"])
    num = ["price","popularity","sentiment"]
    cat = ["category"]
    X, y = df[num+cat], df["google_rating"]
    prep = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
    ])
    models = {
        "RF": RandomForestRegressor(n_estimators=150, random_state=42),
        "GB": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
    }
    best_rmse, best_pipe = np.inf, None
    for m in models.values():
        pipe = Pipeline([("prep", prep), ("reg", m)])
        pipe.fit(X,y)
        rmse = mean_squared_error(y, pipe.predict(X)) ** 0.5
        if rmse < best_rmse:
            best_rmse, best_pipe = rmse, pipe
    best_pipe.fit(X,y)
    df["predicted_rating"] = best_pipe.predict(X).round(1)
    return df

df = train_model(df_raw)

# ───── BEAUTIFIED FILTERS ─────
with st.sidebar:
    # Banner
    st.markdown('<div class="sidebar-banner">'
                '<img src="https://img.icons8.com/emoji/96/000000/chef-emoji.png"/>'
                '<h2>Filters</h2></div>',
                unsafe_allow_html=True)

    # Reset button
    if st.button("🔄 Reset Filters", key="reset"):
        st.session_state.clear()

    # City Expander
    with st.expander("🌆 City"):
        if "city" in df.columns:
            cities = ["All"]+sorted(df["city"].dropna().unique())
            sel_city = st.selectbox("Select City", cities)
        else:
            sel_city = "All"

    # ZIP Expander
    with st.expander("🏷️ ZIP Code"):
        if "postal_code" in df.columns:
            zips = ["All"]+sorted(df["postal_code"].dropna().unique())
            sel_zip = st.selectbox("Select ZIP", zips)
        else:
            sel_zip = "All"

    # Category Expander
    with st.expander("📂 Category"):
        if "category" in df.columns:
            cats = sorted(df["category"].dropna().unique())
            sel_cats = st.multiselect("Choose Categories", cats)
        else:
            sel_cats = []

    # Price Expander
    with st.expander("💸 Price Level"):
        pr = st.slider("Price (1–10)", 1, 10, (1,10))

# Apply filters
df_f = df.copy()
if sel_city != "All":
    df_f = df_f[df_f["city"]==sel_city]
if sel_zip != "All":
    df_f = df_f[df_f["postal_code"]==sel_zip]
if sel_cats:
    df_f = df_f[df_f["category"].isin(sel_cats)]
df_f = df_f[df_f["price"].between(pr[0],pr[1])]

# ───── TOP 5 & METRICS ─────
df_f = df_f.sort_values("predicted_rating", ascending=False)
top5 = df_f.head(5).reset_index(drop=True)

st.title("🍴 Top 5 Restaurants by Predicted Rating")
c1,c2,c3 = st.columns(3)
c1.metric("Matches", len(df_f))
c2.metric("Avg Predicted Rating", f"{df_f['predicted_rating'].mean():.2f}")
c3.metric("Avg Sentiment", f"{df_f['sentiment'].mean():.2f}")

# ───── SELECT & DISPLAY ─────
names = [""]+list(top5["name"])
sel = st.selectbox("Select a restaurant", names)
if not sel:
    st.info("Please select a restaurant."); st.stop()
r = top5[top5["name"]==sel].iloc[0]

st.subheader(f"{sel}")
st.metric("Predicted Rating", f"{r['predicted_rating']} ⭐")

if "formatted_address" in r:
    st.markdown(f"<div class='address'>📍 {r['formatted_address']}</div>", unsafe_allow_html=True)

# Map
if {"latitude","longitude"}.issubset(r.index):
    view = pdk.ViewState(latitude=r["latitude"], longitude=r["longitude"], zoom=14)
    clr = [
        int(255*(1-(r['predicted_rating']-1)/4)),
        int(120+135*(r['predicted_rating']-1)/4),200,180
    ]
    layer = pdk.Layer("ScatterplotLayer", data=pd.DataFrame([r]),
                      get_position='[longitude, latitude]',
                      get_fill_color=clr, get_radius=100)
    st.pydeck_chart(pdk.Deck(initial_view_state=view, layers=[layer]))
else:
    st.error("Location data missing.")

# ───── KEY PHRASES ─────
st.subheader("Key Phrases from Reviews")
raw = r.get("combined_reviews","")
if raw:
    docs = [s.strip() for s in re.split(r'\|\||\n', raw) if s.strip()]
    extra = {"just","really","also","will","get","one"}
    stop_list = list(ENGLISH_STOP_WORDS.union(extra))
    vect = TfidfVectorizer(stop_words=stop_list, ngram_range=(1,2), max_features=50)
    X = vect.fit_transform(docs)
    scores = np.asarray(X.mean(axis=0)).ravel()
    terms = np.array(vect.get_feature_names_out())
    top_terms = terms[np.argsort(scores)[::-1][:8]]
    badges = " ".join(f"<span class='badge'>{w}</span>" for w in top_terms)
    st.markdown(badges, unsafe_allow_html=True)
else:
    st.info("No review text available.")

# ───── SAMPLE REVIEW SNIPPETS ─────
st.subheader("Sample Review Snippets")
if raw:
    for snippet in docs[:3]:
        st.markdown(f"<div class='snippet-card'>“{snippet}”</div>", unsafe_allow_html=True)
else:
    st.info("No reviews to display.")
