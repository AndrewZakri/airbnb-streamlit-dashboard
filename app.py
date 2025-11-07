import os
import math

import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient
import pandas as pd
import numpy as np
import plotly.express as px

# -------------------------------------------------
# Streamlit page setup
# -------------------------------------------------
st.set_page_config(page_title="Airbnb Insights", page_icon="🏠", layout="wide")
load_dotenv()  # Loads .env locally

# -------------------------------------------------
# Config variables (from .env or Streamlit secrets)
# -------------------------------------------------
MONGODB_URI = os.getenv("MONGODB_URI") or st.secrets.get("MONGODB_URI")
DB_NAME = (os.getenv("DB_NAME") or st.secrets.get("DB_NAME")) or "sample_airbnb"
COLL_NAME = (os.getenv("COLL_NAME") or st.secrets.get("COLL_NAME")) or "listingsAndReviews"

if not MONGODB_URI:
    st.error("❌ Missing MONGODB_URI. Add it to your .env or Streamlit Secrets.")
    st.stop()

# -------------------------------------------------
# MongoDB connection (cache_resource = ✅ for non-serializable objects)
# -------------------------------------------------
@st.cache_resource
def get_client():
    return MongoClient(MONGODB_URI, tls=True)

# Optional: upfront connectivity check (shows clear error instead of redaction)
try:
    _ = get_client().admin.command("ping")
except Exception as e:
    st.error("Could not connect to MongoDB. Check firewall/IP allowlist and MONGODB_URI.")
    st.exception(e)
    st.stop()

# -------------------------------------------------
# Fetch helper functions
# -------------------------------------------------
@st.cache_data(show_spinner=True)
def fetch_distincts():
    col = get_client()[DB_NAME][COLL_NAME]
    countries = sorted([c for c in col.distinct("address.country") if isinstance(c, str)])
    markets = sorted([m for m in col.distinct("address.market") if isinstance(m, str)])
    return countries, markets

@st.cache_data(show_spinner=True)
def fetch_price_bounds(country: str | None, market: str | None):
    """
    Returns (min_price:int, max_price:int) with strong type safety.
    Works even if price is string/Decimal128/missing.
    """
    col = get_client()[DB_NAME][COLL_NAME]

    q = {}
    if country:
        q["address.country"] = country
    if market:
        q["address.market"] = market

    pipeline = [
        {"$match": q},
        # Normalize price to double; drop anything non-coercible
        {"$project": {
            "p": {
                "$convert": {"input": "$price", "to": "double", "onError": None, "onNull": None}
            }
        }},
        {"$match": {"p": {"$ne": None}}},
        {"$group": {"_id": None, "minp": {"$min": "$p"}, "maxp": {"$max": "$p"}}}
    ]

    agg = list(col.aggregate(pipeline))
    # If no numeric prices for current filter combo, provide sane defaults
    if not agg or agg[0].get("minp") is None or agg[0].get("maxp") is None:
        return 0, 2000

    # Safe float cast
    try:
        minp_f = float(agg[0]["minp"])
    except Exception:
        minp_f = 0.0
    try:
        maxp_f = float(agg[0]["maxp"])
    except Exception:
        maxp_f = max(minp_f, 1.0)

    # Ensure usable ordering and integers for the slider
    if not (maxp_f > minp_f):
        maxp_f = minp_f + 1.0

    min_i = int(minp_f) if minp_f >= 0 else 0
    max_i = int(maxp_f) if maxp_f >= 1 else (min_i + 1)
    return min_i, max_i

@st.cache_data(show_spinner=True)
def fetch_summary(filters: dict):
    col = get_client()[DB_NAME][COLL_NAME]
    q = {}
    if filters.get("country"):
        q["address.country"] = filters["country"]
    if filters.get("market"):
        q["address.market"] = filters["market"]
    if filters.get("min_bedrooms") is not None:
        q["bedrooms"] = {"$gte": filters["min_bedrooms"]}
    if filters.get("price_range"):
        low, high = filters["price_range"]
        q["price"] = {"$gte": low, "$lte": high}

    proj = {
        "_id": 0,
        "name": 1, "price": 1, "bedrooms": 1, "bathrooms": 1,
        "accommodates": 1, "number_of_reviews": 1,
        "review_scores.review_scores_rating": 1,
        "address.country": 1, "address.market": 1,
        "address.location.coordinates": 1,
        "property_type": 1, "room_type": 1
    }

    docs = list(col.find(q, proj).limit(20000))
    df = pd.json_normalize(docs)

    if "review_scores.review_scores_rating" in df.columns:
        df.rename(columns={"review_scores.review_scores_rating": "rating"}, inplace=True)

    for c in ["price", "bedrooms", "bathrooms", "accommodates", "number_of_reviews", "rating"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "address.location.coordinates" in df.columns:
        coords = df["address.location.coordinates"].dropna().apply(
            lambda x: x if isinstance(x, (list, tuple)) and len(x) == 2 else [np.nan, np.nan]
        )
        df["lon"] = coords.apply(lambda v: v[0] if isinstance(v, (list, tuple)) else np.nan)
        df["lat"] = coords.apply(lambda v: v[1] if isinstance(v, (list, tuple)) else np.nan)

    return df

# -------------------------------------------------
# Sidebar controls
# -------------------------------------------------
st.sidebar.title("Filters")
countries, markets = fetch_distincts()

country = st.sidebar.selectbox("Country", ["(All)"] + countries)
country_val = None if country == "(All)" else country

market_options = ["(All)"] + ([m for m in markets if country_val is None] or markets)
market = st.sidebar.selectbox("Market / City", market_options)
market_val = None if market == "(All)" else market

minp, maxp = fetch_price_bounds(country_val, market_val)

# Defensive slider (avoids errors if bounds collapse)
if maxp <= minp:
    maxp = minp + 1

# Choose safe defaults within [minp, maxp]
low_default = max(minp, min(50, maxp - 1))
high_default = max(low_default + 1, min(maxp, minp + 50))

price_range = st.sidebar.slider(
    "Nightly Price (USD)",
    min_value=minp,
    max_value=maxp,
    value=(low_default, high_default),
    step=5
)

min_bedrooms = st.sidebar.number_input("Min bedrooms", min_value=0, max_value=10, value=0, step=1)

filters = {
    "country": country_val,
    "market": market_val,
    "price_range": price_range,
    "min_bedrooms": int(min_bedrooms)
}

# -------------------------------------------------
# Main dashboard body
# -------------------------------------------------
df = fetch_summary(filters)
st.title("🏠 Airbnb Insights Dashboard")
st.caption("Connected to Azure Cosmos DB for MongoDB (vCore)")

# KPIs
col1, col2, col3, col4 = st.columns(4)
total_listings = int(len(df))
avg_price = float(df["price"].dropna().mean()) if "price" in df else float("nan")
avg_rating = float(df["rating"].dropna().mean()) if "rating" in df else float("nan")
avg_reviews = float(df["number_of_reviews"].dropna().mean()) if "number_of_reviews" in df else float("nan")

col1.metric("Listings", f"{total_listings:,}")
col2.metric("Avg Price", f"${avg_price:,.0f}" if not math.isnan(avg_price) else "—")
col3.metric("Avg Rating", f"{avg_rating:,.1f}" if not math.isnan(avg_rating) else "—")
col4.metric("Avg #Reviews", f"{avg_reviews:,.1f}" if not math.isnan(avg_reviews) else "—")

st.divider()

# Price distribution
if "price" in df.columns and len(df.dropna(subset=["price"])) > 0:
    fig_price = px.histogram(df, x="price", nbins=40, title="Price Distribution")
    fig_price.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=300)
    st.plotly_chart(fig_price, use_container_width=True)

# Rating vs Price
if all(c in df.columns for c in ["price", "rating", "accommodates"]):
    scatter_df = df.dropna(subset=["price", "rating"]).copy()
    if len(scatter_df) > 0:
        fig_scatter = px.scatter(
            scatter_df.sample(min(2000, len(scatter_df))),
            x="rating", y="price", size="accommodates",
            hover_data=["name", "address.market", "bedrooms"],
            title="Price vs. Rating (bubble = accommodates)"
        )
        fig_scatter.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=350)
        st.plotly_chart(fig_scatter, use_container_width=True)

# Market summary
group_cols = [c for c in ["address.market", "address.country"] if c in df.columns]
if group_cols:
    g = (df.dropna(subset=["price"])
           .groupby(group_cols)
           .agg(listings=("name", "count"),
                avg_price=("price", "mean"),
                median_price=("price", "median"),
                avg_rating=("rating", "mean"),
                avg_reviews=("number_of_reviews", "mean"))
           .reset_index()
           .sort_values(by="listings", ascending=False))
    for c in ["avg_price", "median_price"]:
        g[c] = g[c].round(0)
    for c in ["avg_rating", "avg_reviews"]:
        g[c] = g[c].round(1)
    st.subheader("Market Summary")
    st.dataframe(g, use_container_width=True, height=320)

# Map
if all(c in df.columns for c in ["lat", "lon"]):
    geo_df = df.dropna(subset=["lat", "lon"]).copy()
    if len(geo_df) > 0:
        st.subheader("Map of Listings")
        map_sample = geo_df.sample(min(len(geo_df), 3000))
        fig_map = px.scatter_mapbox(
            map_sample,
            lat="lat", lon="lon",
            hover_name="name",
            hover_data={"price": True, "rating": True, "address.market": True, "lat": False, "lon": False},
            zoom=1, height=500
        )
        fig_map.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_map, use_container_width=True)

# Notes
with st.expander("ℹ️ Data & Method Notes"):
    st.markdown("""
- Data: `sample_airbnb.listingsAndReviews` (Cosmos DB for MongoDB vCore)
- Filters run server-side; visuals aggregate client-side.
- Coordinates `[lon, lat]` converted to `lat/lon` for map.
""")
