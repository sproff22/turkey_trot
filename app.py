from flask import Flask, render_template_string
import pandas as pd
import requests
import geopandas as gpd
from shapely.geometry import Point
import folium
import numpy as np
from scipy.spatial import cKDTree
from datetime import datetime
import branca.colormap as cm
import os

api_key = os.environ.get("EBIRD_API_KEY")

app = Flask(__name__)

@app.route("/")
def turkey_map():
    # --- Your data fetching and processing code here ---
    species_code = "wiltur"
    region_code = "US-MA"
    url = f"https://api.ebird.org/v2/data/obs/{region_code}/recent/{species_code}"
    headers = {"X-eBirdApiToken": api_key}
    params = {"maxResults": 10000, "back": 30}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        return f"Error: {response.status_code} {response.text}"

    data = response.json()
    df = pd.DataFrame(data)
    df['obsDt'] = pd.to_datetime(df['obsDt'], format='mixed')
    geometry = [Point(xy) for xy in zip(df['lng'], df['lat'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    turkey_gdf = gdf.copy()
    gdf_m = turkey_gdf.to_crs(epsg=3857).copy()
    coords = np.array(list(zip(gdf_m.geometry.x, gdf_m.geometry.y)))
    tree = cKDTree(coords)
    turkey_gdf['howMany'] = turkey_gdf['howMany'].fillna(1)
    howmany_vals = turkey_gdf['howMany'].clip(upper=turkey_gdf['howMany'].quantile(0.99))
    howmany_norm = howmany_vals.rank(pct=True).values
    turkey_gdf['obsDt'] = pd.to_datetime(turkey_gdf['obsDt'])
    now = datetime.utcnow()
    days_since = (now - turkey_gdf['obsDt']).dt.days
    recency_score = 1 - (days_since / days_since.max())
    scores = []
    for i, (x, y) in enumerate(coords):
        indices = tree.query_ball_point([x, y], r=1609.34)
        score = 0
        for j in indices:
            dist = np.linalg.norm(coords[i] - coords[j])
            decay = max(0, 1 - (dist / 1609.34))
            count = turkey_gdf.iloc[j]['howMany']
            score += count * decay
        scores.append(score)
    gdf_m['turkey_score'] = scores
    gdf_m['turkey_score_norm'] = gdf_m['turkey_score'] / max(scores)
    gdf_m['howmany_norm'] = howmany_norm
    gdf_m['recency_score'] = recency_score.values
    gdf_latlon = gdf_m.to_crs(epsg=4326)
    colormap = cm.linear.YlOrRd_09.scale(0, 1)
    center_lat = gdf_latlon.geometry.y.mean()
    center_lon = gdf_latlon.geometry.x.mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles=None)
    folium.TileLayer(
        tiles='http://mt1.google.com/vt/lyrs=r&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Maps',
        overlay=False,
        control=True
    ).add_to(m)
    for _, row in gdf_latlon.iterrows():
        loc = [row.geometry.y, row.geometry.x]
        base_color = colormap(row['howmany_norm'])
        base_opacity = 0.1 + 0.4 * row['recency_score']
        date = pd.to_datetime(row["obsDt"]).strftime("%b %d, %Y")
        popup_text = f"{row['locName']}<br>{(row['howMany'])} turkey(s)<br>{date}"
        for i in range(5):
            step_radius = 300 + i * 200
            step_opacity = base_opacity * (1 - i / 5)
            folium.Circle(
                location=loc,
                radius=step_radius,
                color=None,
                fill=True,
                fill_color=base_color,
                fill_opacity=step_opacity
            ).add_to(m)
        folium.CircleMarker(
            location=loc,
            radius=2,
            color='green',
            fill=True,
            fill_opacity=1.0,
            popup=folium.Popup(popup_text, max_width=500)
        ).add_to(m)
    map_html = m._repr_html_()
    return render_template_string("""
        <html>
        <head><title>Turkey Heatmap</title></head>
        <body>
            <h1>Live Turkey Heatmap</h1>
            {{ map_html|safe }}
        </body>
        </html>
    """, map_html=map_html)

if __name__ == "__main__":
    app.run(debug=True)