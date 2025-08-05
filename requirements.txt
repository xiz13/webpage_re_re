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

# ───── PAGE & GLOBAL CSS ─────
st.set_page_config("Restaurant Finder", "🍽️", layout="wide")
st.markdown("""
<style>
/* Sidebar card */
[data-testid="stSidebar"] > div:first-child {
  background: #fff;
  padding: 16px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
/* Filter labels */
.stSidebar label {
  font-weight: 600;
  margin-top: 8px;
}
/* Badges */
.badge {display:inline-block; background:#ffb300; color:#fff; border-radius:4px; padding:4px 8px; margin:4px 4px 4px 0; font-size:0.85rem;}
/* Review snippet cards */
.snippet-card {
  background: #fff;
  border-left: 4px solid #43a047;
  padding: 12px 16px;
  margin: 8px 0;
  border-radius: 6px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.1);
}
/* Formatted address */
.address {font-style:italic; color:#555; margin-bottom:12px;}
</style>
""", unsafe_allow_html=True)

# ───── LOAD DATA ─────
DATA_URL = "https://drive.google.com/uc?export=download&id=1iRWeGaDybybQ2eiTCyEgyXDYbH5FpFup"
@st.cache_data
def load_data(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=60); r.raise_for_status()
    return pd.read_csv(BytesIO(r.content))
df_raw = load_data(DATA_URL)

# ───── TRAIN MODEL ─────
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
        pipe = Pipeline([("prep", prep),("reg", m)])
        pipe.fit(X,y)
        rmse = mean_squared_error(y, pipe.predict(X), squared=False)
        if rmse < best_rmse:
            best_rmse, best_pipe = rmse, pipe
    best_pipe.fit(X,y)
    df["predicted_rating"] = best_pipe.predict(X).round(1)
    return df

df = train_model(df_raw)

# ───── FILTER SECTION ─────
st.sidebar.header("Filters")

# City
if "city" in df.columns:
    cities = ["All"]+sorted(df["city"].dropna().unique())
    city = st.sidebar.selectbox("City", cities)
    if city!="All": df = df[df["city"]==city]

# ZIP
if "postal_code" in df.columns:
    zips = ["All"]+sorted(df["postal_code"].dropna().unique())
    zp = st.sidebar.selectbox("ZIP code", zips)
    if zp!="All": df = df[df["postal_code"]==zp]

# Category
if "category" in df.columns:
    cats = sorted(df["category"].dropna().unique())
    sel = st.sidebar.multiselect("Category", cats)
    if sel: df = df[df["category"].isin(sel)]

# Price 1–10
pmin,pmax=1,10
pr = st.sidebar.slider("Price level", pmin, pmax, (pmin,pmax))
df = df[df["price"].between(pr[0], pr[1])]

# ───── TOP 5 CALC ─────
df = df.sort_values("predicted_rating", ascending=False)
top5 = df.head(5).reset_index(drop=True)

st.title("🍴 Top 5 Restaurants by Predicted Rating")
c1,c2,c3 = st.columns(3)
c1.metric("Matches", len(df))
c2.metric("Avg Predicted Rating", f"{df['predicted_rating'].mean():.2f}")
c3.metric("Avg Sentiment", f"{df['sentiment'].mean():.2f}")

# ───── RESTAURANT SELECT ─────
names = [""] + list(top5["name"])
sel = st.selectbox("Select a restaurant to inspect", names)
if not sel:
    st.info("Please select a restaurant above."); st.stop()
r = top5[top5["name"]==sel].iloc[0]

# ───── DISPLAY PREDICTED RATING ─────
st.subheader(f"{sel}")
st.metric("Predicted Rating", f"{r['predicted_rating']} ⭐")

# ───── FORMATTED ADDRESS ─────
if "formatted_address" in r:
    st.markdown(f"<div class='address'>📍 {r['formatted_address']}</div>", unsafe_allow_html=True)

# ───── MAP ─────
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

# ───── KEY PHRASE EXTRACTION ─────
st.subheader("Key Phrases from Reviews")
raw = r.get("combined_reviews","")
if raw:
    # split & clean
    docs = [s.strip() for s in re.split(r'\|\||\n', raw) if s.strip()]
    # build extended stopwords
    extra = {"just","really","also","will","get","one"}
    stops = ENGLISH_STOP_WORDS.union(extra)
    vect = TfidfVectorizer(stop_words=stops, ngram_range=(1,2), max_features=50)
    X = vect.fit_transform(docs)
    scores = np.asarray(X.mean(axis=0)).ravel()
    terms = np.array(vect.get_feature_names_out())
    top_terms = terms[np.argsort(scores)[::-1][:8]]
    badges = " ".join(f"<span class='badge'>{w}</span>" for w in top_terms)
    st.markdown(badges, unsafe_allow_html=True)
else:
    st.info("No review text available.")

# ───── SAMPLE REVIEW SNIPPETS ─────
st.subheader("Sample Reviews")
if raw:
    for snippet in docs[:3]:
        st.markdown(f"<div class='snippet-card'>“{snippet}”</div>", unsafe_allow_html=True)
else:
    st.info("No reviews to display.")
