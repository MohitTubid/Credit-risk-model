"""
Precompute per-US-state aggregates -> state_stats.csv for the pydeck map on the Insights
page. Like feature_defaults.json, this ships only AGGREGATES (no raw rows), so the map
works on the deployed app without publishing the credit data.

Run:  python make_state_stats.py   (owner, local; re-run after data changes)
"""
import os
import pandas as pd
from ml_eval import DATA_PATH, REG_DATA_PATH

HERE = os.path.dirname(os.path.abspath(__file__))

# Approximate geographic centroid (lat, lon) per US state, for placing map columns.
STATE_CENTROIDS = {
    "AL": (32.81, -86.79), "AK": (61.37, -152.40), "AZ": (33.73, -111.43),
    "AR": (34.97, -92.37), "CA": (36.12, -119.68), "CO": (39.06, -105.31),
    "CT": (41.60, -72.76), "DE": (39.32, -75.51), "FL": (27.77, -81.69),
    "GA": (33.04, -83.64), "HI": (21.09, -157.50), "ID": (44.24, -114.48),
    "IL": (40.35, -88.99), "IN": (39.85, -86.26), "IA": (42.01, -93.21),
    "KS": (38.53, -96.73), "KY": (37.67, -84.67), "LA": (31.17, -91.87),
    "ME": (44.69, -69.38), "MD": (39.06, -76.80), "MA": (42.23, -71.53),
    "MI": (43.33, -84.54), "MN": (45.69, -93.90), "MS": (32.74, -89.68),
    "MO": (38.46, -92.29), "MT": (46.92, -110.45), "NE": (41.13, -98.27),
    "NV": (38.31, -117.06), "NH": (43.45, -71.56), "NJ": (40.30, -74.52),
    "NM": (34.84, -106.25), "NY": (42.17, -74.95), "NC": (35.63, -79.81),
    "ND": (47.53, -99.78), "OH": (40.39, -82.76), "OK": (35.57, -96.93),
    "OR": (44.57, -122.07), "PA": (40.59, -77.21), "RI": (41.68, -71.51),
    "SC": (33.86, -80.95), "SD": (44.30, -99.44), "TN": (35.75, -86.69),
    "TX": (31.05, -97.56), "UT": (40.15, -111.86), "VT": (44.05, -72.71),
    "VA": (37.77, -78.17), "WA": (47.40, -121.49), "WV": (38.49, -80.95),
    "WI": (44.27, -89.62), "WY": (42.76, -107.30), "DC": (38.90, -77.03),
}

COL = "state_orig_time"

clf = pd.read_csv(DATA_PATH)
agg = (clf.groupby(COL)
          .agg(n=("default_time", "size"), default_rate=("default_time", "mean"))
          .reset_index())

reg = pd.read_csv(REG_DATA_PATH)
lgd = reg.groupby(COL)["lgd_time"].mean().rename("avg_lgd").reset_index()

df = agg.merge(lgd, on=COL, how="left")
df["lat"] = df[COL].map(lambda s: STATE_CENTROIDS.get(s, (None, None))[0])
df["lon"] = df[COL].map(lambda s: STATE_CENTROIDS.get(s, (None, None))[1])
df = df.dropna(subset=["lat", "lon"]).rename(columns={COL: "state"})

# Rounded display columns
df["default_rate"] = df["default_rate"].round(4)
df["default_pct"] = (df["default_rate"] * 100).round(2)
df["avg_lgd"] = df["avg_lgd"].round(3)

out = os.path.join(HERE, "state_stats.csv")
df.to_csv(out, index=False)
print(f"Saved {len(df)} states -> {out}")
print(df.sort_values("default_rate", ascending=False).head(10).to_string(index=False))
