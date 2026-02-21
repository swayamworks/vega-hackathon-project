import folium
import pandas as pd
from folium.plugins import HeatMap

# =====================================================
# STEP 1: LOAD MODEL OUTPUT
# =====================================================

df = pd.read_csv("final_dashboard_dataset_hackathon.csv")

required_cols = ['latitude', 'longitude', 'cluster_id', 'risk_score', 'risk_level']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

print("Loaded model output:", df.shape)

# =====================================================
# STEP 1.5: FIX & CLEAN COORDINATES
# =====================================================

# Drop nulls
df = df.dropna(subset=["latitude", "longitude"])

# Auto-fix swapped lat/lon (if latitude looks like longitude)
if df["latitude"].mean() > 50:
    print("Swapping latitude and longitude...")
    df[["latitude", "longitude"]] = df[["longitude", "latitude"]]

# Remove impossible global coordinates
df = df[
    (df["latitude"].between(-90, 90)) &
    (df["longitude"].between(-180, 180))
]

# Restrict to India bounding box
df = df[
    (df["latitude"].between(6, 38)) &
    (df["longitude"].between(68, 98))
]

# Remove obvious Arabian Sea spillover (west coast correction)
df = df[
    ~(
        (df["latitude"].between(15, 22)) &
        (df["longitude"] < 72.7)
    )
]

print("After cleaning:", df.shape)

# =====================================================
# STEP 2: BASIC ANALYTICS
# =====================================================

total_incidents = len(df)
critical_pct = round((df['risk_level'] == "Critical").mean() * 100, 2)

cluster_risk = (
    df[df['cluster_id'] != -1]
    .groupby("cluster_id")['risk_score']
    .mean()
    .sort_values(ascending=False)
)

top_clusters = cluster_risk.head(5)

# =====================================================
# STEP 3: MAP WITH DARK DEFAULT + LIGHT TOGGLE
# =====================================================

m = folium.Map(
    location=[22.5937, 78.9629],
    zoom_start=5,
    tiles=None
)

folium.TileLayer(
    tiles="CartoDB dark_matter",
    name="Dark Mode",
    attr="© OpenStreetMap contributors © CARTO"
).add_to(m)

folium.TileLayer(
    tiles="CartoDB positron",
    name="Light Mode",
    attr="© OpenStreetMap contributors © CARTO"
).add_to(m)

# =====================================================
# STEP 4: FEATURE GROUPS
# =====================================================

low_group = folium.FeatureGroup(name="Low Risk")
moderate_group = folium.FeatureGroup(name="Moderate Risk")
critical_group = folium.FeatureGroup(name="Critical Risk")

# =====================================================
# STEP 5: INCIDENT MARKERS
# =====================================================

for _, row in df.iterrows():

    popup_text = f"""
    <b>Cluster:</b> {row['cluster_id']}<br>
    <b>Risk Score:</b> {row['risk_score']:.2f}<br>
    <b>Risk Level:</b> {row['risk_level']}
    """

    marker = folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=5,
        fill=True,
        fill_opacity=0.8,
        popup=popup_text
    )

    if row["risk_level"] == "Low":
        marker.options.update(color="green", fillColor="green")
        marker.add_to(low_group)
    elif row["risk_level"] == "Moderate":
        marker.options.update(color="orange", fillColor="orange")
        marker.add_to(moderate_group)
    else:
        marker.options.update(color="red", fillColor="red")
        marker.add_to(critical_group)

low_group.add_to(m)
moderate_group.add_to(m)
critical_group.add_to(m)

# =====================================================
# STEP 6: HEATMAP
# =====================================================

heat_data = df[["latitude", "longitude", "risk_score"]].values.tolist()

HeatMap(
    heat_data,
    radius=18,
    blur=25,
    min_opacity=0.4
).add_to(m)

# =====================================================
# STEP 7: CLUSTER CENTER BLUE PIN MARKERS
# =====================================================

cluster_centers = (
    df[df["cluster_id"] != -1]
    .groupby("cluster_id")[["latitude", "longitude"]]
    .mean()
)

for cid, row in cluster_centers.iterrows():
    folium.Marker(
        location=[row["latitude"], row["longitude"]],
        icon=folium.Icon(
            color="blue",
            icon="info-sign",
            prefix="glyphicon"
        ),
        popup=f"<b>Cluster {cid}</b><br>Avg Risk: {cluster_risk[cid]:.2f}"
    ).add_to(m)

# =====================================================
# STEP 8: KPI PANEL
# =====================================================

kpi_html = f"""
<div style="
position: fixed;
top: 20px;
left: 20px;
width: 280px;
background-color: rgba(0,0,0,0.85);
color: white;
border-radius: 10px;
padding: 15px;
z-index:9999;
font-size:14px;">
<h4>🚦 System Intelligence Overview</h4>
Total Incidents: <b>{total_incidents}</b><br>
Critical Risk %: <b>{critical_pct}%</b><br>
Highest Risk Cluster: <b>{top_clusters.index[0] if len(top_clusters)>0 else "N/A"}</b><br>
</div>
"""

m.get_root().html.add_child(folium.Element(kpi_html))

# =====================================================
# STEP 9: TOP 5 DANGEROUS CLUSTERS PANEL
# =====================================================

cluster_text = "<br>".join(
    [f"Cluster {cid}: Avg Risk {score:.2f}"
     for cid, score in top_clusters.items()]
)

danger_html = f"""
<div style="
position: fixed;
top: 200px;
left: 20px;
width: 280px;
background-color: rgba(0,0,0,0.85);
color: white;
border-radius: 10px;
padding: 15px;
z-index:9999;
font-size:14px;">
<h4>🔥 Top 5 High-Risk Zones</h4>
{cluster_text}
</div>
"""

m.get_root().html.add_child(folium.Element(danger_html))

# =====================================================
# STEP 10: LEGEND
# =====================================================

legend_html = """
<div style="
position: fixed;
bottom: 40px;
left: 20px;
width: 200px;
background-color: white;
border-radius: 8px;
padding: 10px;
z-index:9999;
font-size:14px;">
<b>Risk Legend</b><br>
🟢 Low<br>
🟠 Moderate<br>
🔴 Critical
</div>
"""

m.get_root().html.add_child(folium.Element(legend_html))

folium.LayerControl(collapsed=False).add_to(m)

m.save("AI_Model_Risk_Map_Pro.html")

print("🚀 Professional Hackathon Dashboard Generated Successfully.")