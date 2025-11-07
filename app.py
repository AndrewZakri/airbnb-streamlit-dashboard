import os
import math
from datetime import datetime
from functools import lru_cache

import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient
import pandas as pd
import numpy as np
import plotly.express as px

# ----------------------
# Config & connection
# ----------------------
st.set_page_config(page_title="Airbnb Insights", page_icon="🏠", layout="wide")
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "sample_airbnb")
COLL_NAME = os.getenv("COLL_NAME", "listingsAndReviews")

@st.cache_data(show_spinner=False)
def get_client():
    client = MongoClient(MONGODB_URI, tls=True)
    return client

@st.cache_data(show_spinner=True)
def fetch_distincts():
    col = get_client()[DB_NAME][COLL_NAME]
    countries = sorted([c for c in col.distinct("address.country") if isinstance(c, str)])
    markets = sorted([m for m in col.distinct("address.market") if isinstance(m, str)])
    cuisines = []  # placeholder if you extend schema later
    return countries, markets

@st.cache_data(show_spinner=True)
def fetch_summary(filters: dict):
    """
    Pull lean fields only; keep payload small for speed.
    """
    col = get_client()[DB_NAME][COLL_NAME]

    query = {}
    if filters.get("country"):
        query["address.country"] = filters["country"]
    if filters.get("market"):
        query["address.market"] = filters["market"]
    if filters.get("min_bedrooms") is not None:
        query["bedrooms"] = {"$gte": filters["min_bedrooms"]}
    if filters.get("price_range"):
        low, high = filters["price_range"]
        query["price"] = {"$gte": low, "$lte": high}

    projection = {
        "_id": 0,
        "name": 1,
        "price": 1,
        "bedrooms": 1,
        "bathrooms": 1,
        "accommodates": 1,
        "number_of_reviews": 1,
        "review_scores.review_scores_rating": 1,
        "address.country": 1,
        "address.market": 1,
        "address.location.coordinates": 1,
        "property_type": 1,
        "room_type": 1,
    }

    cursor = col.find(query, projection).limit(20000)  # safety cap for client-side rendering
    docs = list(cursor)

    # Normalize
    df = pd.json_normalize(docs)
    # rename nested fields
    if "review_scores.review_scores_rating" in df.columns:
        df.rename(columns={"review_scores.review_scores_rating": "rating"}, inplace=True)
    # numeric safety
    for c in ["price", "bedrooms", "bathrooms", "accommodates", "number_of_reviews", "rating"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # coordinates -> lat/lon
    if "address.location.coordinates" in df.columns:
        coords = df["address.location.coordinates"].dropna().apply(
            lambda x: x if isinstance(x, (list, tuple)) and len(x) == 2 else [np.nan, np.nan]
        )
        df["lon"] = coords.apply(lambda v: v[0] if isinstance(v, (list, tuple)) else np.nan)
        df["lat"] = coords.apply(lambda v: v[1] if isinstance(v, (list, tuple)) else np.nan)

    return df

@st.cache_data(show_spinner=True)
def fetch_price_bounds(country: str | None, market: str | None):
    """
    Get min/max price quickly for slider bounds (fall back to sample if empty).
    """
    col = get_client()[DB_NAME][COLL_NAME]
    q = {}
    if country:
        q["address.country"] = country
    if market:
        q["address.market"] = market

    pipeline = [
        {"$match": q},
        {"$match": {"price": {"$type": "number"}}},
        {"$group": {"_id": None, "minp": {"$min": "$price"}, "maxp": {"$max": "$price"}}}
    ]
    agg = list(col.aggregate(pipeline))
    if agg:
        minp = int(max(0, math.floor(agg[0]["minp"])))
        maxp = int(math.ceil(agg[0]["maxp"]))
        return minp, maxp
    return 0, 2000

# ----------------------
# Sidebar controls
# ----------------------
st.sidebar.title("Filters")
countries, markets = fetch_distincts()

country = st.sidebar.selectbox("Country", options=["(All)"] + countries)
country_val = None if country == "(All)" else country

market_options = ["(All)"] + ([m for m in markets if country_val is None] or markets)
market = st.sidebar.selectbox("Market / City", options=market_options)
market_val = None if market == "(All)" else market

minp, maxp = fetch_price_bounds(country_val, market_val)
price_range = st.sidebar.slider("Nightly Price (USD)", min_value=minp, max_value=maxp,
                                value=(min(minp, 50), min(maxp, 500)), step=5)

min_bedrooms = st.sidebar.number_input("Min bedrooms", min_value=0, max_value=10, value=0, step=1)

filters = {
    "country": country_val,
    "market": market_val,
    "price_range": price_range,
    "min_bedrooms": int(min_bedrooms)
}

# ----------------------
# Data pull
# ----------------------
df = fetch_summary(filters)

st.title("🏠 Airbnb Insights Dashboard")
st.caption("Backed by Azure Cosmos DB for MongoDB (vCore) • Interactive KPIs, trends, and map")

# ----------------------
# KPIs
# ----------------------
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

# ----------------------
# Price distribution
# ----------------------
if "price" in df.columns and len(df.dropna(subset=["price"])) > 0:
    fig_price = px.histogram(df, x="price", nbins=40, title="Price Distribution")
    fig_price.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=300)
    st.plotly_chart(fig_price, use_container_width=True)

# ----------------------
# Rating vs. Price (bubble by accommodates)
# ----------------------
if all(c in df.columns for c in ["price", "rating", "accommodates"]):
    scatter_df = df.dropna(subset=["price", "rating"]).copy()
    if len(scatter_df) > 0:
        fig_scatter = px.scatter(
            scatter_df.sample(min(2000, len(scatter_df))),  # limit for responsiveness
            x="rating", y="price", size="accommodates", hover_data=["name", "address.market", "bedrooms"],
            title="Price vs. Rating (bubble=size of accommodates)"
        )
        fig_scatter.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=350)
        st.plotly_chart(fig_scatter, use_container_width=True)

# ----------------------
# Market-level summary table
# ----------------------
group_cols = []
if "address.market" in df.columns:
    group_cols.append("address.market")
if "address.country" in df.columns:
    group_cols.append("address.country")

if group_cols:
    g = (df
         .dropna(subset=["price"])
         .groupby(group_cols)
         .agg(
            listings=("name", "count"),
            avg_price=("price", "mean"),
            median_price=("price", "median"),
            avg_rating=("rating", "mean"),
            avg_reviews=("number_of_reviews", "mean")
         )
         .reset_index()
         .sort_values(by="listings", ascending=False))
    g["avg_price"] = g["avg_price"].round(0)
    g["median_price"] = g["median_price"].round(0)
    g["avg_rating"] = g["avg_rating"].round(1)
    g["avg_reviews"] = g["avg_reviews"].round(1)

    st.subheader("Market Summary")
    st.dataframe(g, use_container_width=True, height=320)

# ----------------------
# Map
# ----------------------
if all(c in df.columns for c in ["lat", "lon"]):
    geo_df = df.dropna(subset=["lat", "lon"]).copy()
    if len(geo_df) > 0:
        st.subheader("Map of Listings")
        # limit for speed
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

# ----------------------
# Notes
# ----------------------
with st.expander("ℹ️ Data & Method Notes"):
    st.markdown("""
- Data source: `sample_airbnb.listingsAndReviews` (Cosmos DB for MongoDB, vCore).
- Filters apply server-side (MongoDB query) for efficiency, then results are summarized client-side.
- Coordinates from `address.location.coordinates` are `[lon, lat]` in the source; the app converts them to `lat/lon`.
- For larger deployments, consider using MongoDB `$facet` / `$group` pipelines server-side to pre-aggregate metrics.
""")
