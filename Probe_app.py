import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
import streamlit as st
import streamlit.components.v1 as components
from sklearn.cluster import DBSCAN

plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 400

st.set_page_config(page_title="Lake Profile Analyzer", layout="wide")
st.title("Lake Water-Quality Profile Analyzer")

DATE_COL = "Date"
TIME_COL = "Time"
DEP_COL = "DEP m"
LAT_COL = "Lat"
LON_COL = "Lon"

STATION_CLUSTER_RADIUS_M = 60
MIN_CLUSTER_POINTS = 10
MIN_PROFILE_DEPTH_M = 0.5
MATCH_RADIUS_BETWEEN_DATES_M = 60
MAP_ZOOM = 15

VALUE_COLS = {
    "DO %": "DO %",
    "DO mg/L": "DO mg/L",
    "SPC-uS/cm": "SPC-uS/cm",
    "pH": "pH",
    "ORP mV": "ORP mV",
    "Chl ug/L": "Chl ug/L",
    "PC ug/L": "PC ug/L",
    "PC / Chl": "PC_Chl_ratio",
}

BASE_COLS = [
    DATE_COL, TIME_COL, DEP_COL, LAT_COL, LON_COL,
    "DO %", "DO mg/L", "SPC-uS/cm", "pH", "ORP mV", "Chl ug/L", "PC ug/L"
]


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=400, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf


def clean_filename(name):
    return str(name).replace("/", "_").replace("\\", "_").replace(" ", "_")


def process_file(uploaded_file):
    df = pd.read_excel(uploaded_file)

    missing = [c for c in BASE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {uploaded_file.name}: {missing}")

    df = df[BASE_COLS].copy()

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL].astype(str), format="%H:%M:%S", errors="coerce")

    for col in [DEP_COL, LAT_COL, LON_COL, "DO %", "DO mg/L", "SPC-uS/cm",
                "pH", "ORP mV", "Chl ug/L", "PC ug/L"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["PC_Chl_ratio"] = df["PC ug/L"] / df["Chl ug/L"]
    df.loc[df["Chl ug/L"] == 0, "PC_Chl_ratio"] = pd.NA

    df = df.dropna(subset=[DATE_COL, TIME_COL, DEP_COL, LAT_COL, LON_COL]).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No valid rows found in {uploaded_file.name}")

    file_name = Path(uploaded_file.name).stem
    df["file_name"] = file_name
    df["date"] = df[DATE_COL].dt.date.astype(str)

    df["time_str"] = df[TIME_COL].dt.strftime("%H:%M:%S")
    df["start_time"] = df[TIME_COL].min().strftime("%H:%M")
    df["end_time"] = df[TIME_COL].max().strftime("%H:%M")
    df["time_range"] = df["start_time"] + " - " + df["end_time"]
    df["datetime_label"] = df[DATE_COL].dt.strftime("%d/%m/%Y") + "  " + df["time_range"]

    # ------------------------------------------------
    # GPS-based station clustering
    # ------------------------------------------------
    lat0 = df[LAT_COL].median()
    lon0 = df[LON_COL].median()

    x_m = (df[LON_COL] - lon0) * 111320 * np.cos(np.deg2rad(lat0))
    y_m = (df[LAT_COL] - lat0) * 111320
    coords_m = np.column_stack([x_m, y_m])

    clustering = DBSCAN(
        eps=STATION_CLUSTER_RADIUS_M,
        min_samples=MIN_CLUSTER_POINTS
    ).fit(coords_m)

    df["location_id"] = clustering.labels_

    # remove GPS noise points
    df = df[df["location_id"] != -1].copy()

    if df.empty:
        raise ValueError(f"No GPS station clusters found in {uploaded_file.name}")

    # keep only clusters with enough depth
    good = []

    for loc, g in df.groupby("location_id"):
        if len(g) >= MIN_CLUSTER_POINTS and g[DEP_COL].max() >= MIN_PROFILE_DEPTH_M:
            good.append(loc)

    df = df[df["location_id"].isin(good)].copy()

    if df.empty:
        raise ValueError(f"No valid depth profiles found after GPS clustering in {uploaded_file.name}")

    # renumber stations inside file
    old_to_new = {old: i for i, old in enumerate(sorted(df["location_id"].unique()))}
    df["location_id"] = df["location_id"].map(old_to_new).astype(int)
    df["station"] = "Station_" + (df["location_id"] + 1).astype(str)
    df["location_name"] = df["station"]

    return df


def calculate_means(df):
    rows = []

    for (file_name, date, station), g in df.groupby(["file_name", "date", "station"]):
        loc_id = int(g["location_id"].iloc[0])
        max_depth = int(g[DEP_COL].max())

        for meter in range(0, max_depth + 1):
            if meter == 0:
                w = g[(g[DEP_COL] >= 0) & (g[DEP_COL] <= 0.25)]
            else:
                w = g[(g[DEP_COL] >= meter - 0.25) & (g[DEP_COL] <= meter + 0.25)]

            if len(w):
                row = {
                    "file_name": file_name,
                    "date": date,
                    "station": station,
                    "location_name": station,
                    "location_id": loc_id,
                    "depth_meter": meter,
                    "n_points": len(w),
                    "mean_lat": w[LAT_COL].mean(),
                    "mean_lon": w[LON_COL].mean(),
                    "time_range": w["time_range"].iloc[0],
                    "datetime_label": w["datetime_label"].iloc[0],
                }

                for _, col in VALUE_COLS.items():
                    row[f"mean_{col}"] = w[col].mean()

                rows.append(row)

    for (file_name, date), g in df.groupby(["file_name", "date"]):
        max_depth = int(g[DEP_COL].max())

        for meter in range(0, max_depth + 1):
            if meter == 0:
                w = g[(g[DEP_COL] >= 0) & (g[DEP_COL] <= 0.25)]
            else:
                w = g[(g[DEP_COL] >= meter - 0.25) & (g[DEP_COL] <= meter + 0.25)]

            if len(w):
                row = {
                    "file_name": file_name,
                    "date": date,
                    "station": "Lake mean",
                    "location_name": "Lake mean",
                    "location_id": -1,
                    "depth_meter": meter,
                    "n_points": len(w),
                    "mean_lat": w[LAT_COL].mean(),
                    "mean_lon": w[LON_COL].mean(),
                    "time_range": w["time_range"].iloc[0],
                    "datetime_label": w["datetime_label"].iloc[0],
                }

                for _, col in VALUE_COLS.items():
                    row[f"mean_{col}"] = w[col].mean()

                rows.append(row)

    return pd.DataFrame(rows)


def assign_global_station_names(raw_df, mean_df, radius_m=MATCH_RADIUS_BETWEEN_DATES_M):
    centers = (
        raw_df.groupby(["file_name", "date", "location_id"])
        .agg(mean_lat=(LAT_COL, "mean"), mean_lon=(LON_COL, "mean"))
        .reset_index()
    )

    global_stations = []
    names = {}
    next_id = 1

    for _, r in centers.iterrows():
        assigned = None

        for s in global_stations:
            d = (
                ((r["mean_lat"] - s["lat"]) * 111320) ** 2 +
                ((r["mean_lon"] - s["lon"]) * 111320) ** 2
            ) ** 0.5

            if d <= radius_m:
                assigned = s["name"]
                break

        if assigned is None:
            assigned = f"Station_{next_id}"
            global_stations.append({
                "name": assigned,
                "lat": r["mean_lat"],
                "lon": r["mean_lon"]
            })
            next_id += 1

        names[(r["file_name"], r["location_id"])] = assigned

    raw_df["station"] = raw_df.apply(
        lambda r: names[(r["file_name"], r["location_id"])],
        axis=1
    )
    raw_df["location_name"] = raw_df["station"]

    mean_df["station"] = mean_df.apply(
        lambda r: "Lake mean" if r["location_id"] == -1 else names.get((r["file_name"], r["location_id"]), "Unknown"),
        axis=1
    )
    mean_df["location_name"] = mean_df["station"]

    return raw_df, mean_df


def style_profile(ax, title, xlabel):
    ax.invert_yaxis()
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Depth (m)")

    ymin, ymax = ax.get_ylim()
    max_depth = int(max(ymin, ymax))
    ax.set_yticks(range(0, max_depth + 1))

    ax.grid(True, linestyle="--", alpha=0.35)


def plot_all_variables_for_file(raw_file, mean_file, file_name):
    fig, axes = plt.subplots(1, len(VALUE_COLS), figsize=(30, 5), sharey=False)

    for ax, (label, col) in zip(axes, VALUE_COLS.items()):
        mean_col = f"mean_{col}"

        for station, g in raw_file.groupby("station"):
            ax.scatter(g[col], g[DEP_COL], s=14, alpha=0.20)

            mg = mean_file[mean_file["station"] == station].sort_values("depth_meter")
            ax.plot(mg[mean_col], mg["depth_meter"], marker="o", linewidth=2, label=station)

        style_profile(ax, label, label)

    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(3.7, -0.18),
        ncol=5,
        frameon=False
    )

    time_label = raw_file["datetime_label"].iloc[0]

    fig.suptitle(
        f"{file_name}\n{time_label}",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )

    fig.subplots_adjust(bottom=0.25, top=0.72, wspace=0.35)

    return fig


def plot_comparison(mean_df):
    lake = mean_df[mean_df["station"] == "Lake mean"]

    fig, axes = plt.subplots(1, len(VALUE_COLS), figsize=(30, 5), sharey=False)

    for ax, (label, col) in zip(axes, VALUE_COLS.items()):
        mean_col = f"mean_{col}"

        for key, g in lake.groupby(["date", "file_name"]):
            date, file_name = key
            g = g.sort_values("depth_meter")
            time_range = g["time_range"].iloc[0] if "time_range" in g.columns else ""

            ax.plot(
                g[mean_col],
                g["depth_meter"],
                marker="o",
                linewidth=2,
                label=f"{date} ({time_range})"
            )

        style_profile(ax, label, f"Mean {label}")

    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(3.7, -0.18),
        ncol=4,
        frameon=False
    )

    fig.suptitle("Lake Mean Comparison Between Data", fontsize=16, fontweight="bold")
    fig.subplots_adjust(bottom=0.25, top=0.85)

    return fig


def create_map(df_file, zoom=MAP_ZOOM):
    m = folium.Map(
        location=[df_file[LAT_COL].mean(), df_file[LON_COL].mean()],
        zoom_start=zoom,
        tiles="Esri.WorldImagery"
    )

    colors = ["blue", "red", "green", "purple", "orange", "black", "cadetblue"]

    for i, (station, g) in enumerate(df_file.groupby("station")):
        color = colors[i % len(colors)]

        folium.Marker(
            location=[g[LAT_COL].mean(), g[LON_COL].mean()],
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    font-size:14px;
                    font-weight:bold;
                    color:white;
                    background:#1f4e79;
                    border-radius:6px;
                    padding:4px 8px;
                    border:2px solid white;
                    white-space:nowrap;">
                    {station}
                </div>
                """
            )
        ).add_to(m)

        for _, r in g.iterrows():
            folium.CircleMarker(
                location=[r[LAT_COL], r[LON_COL]],
                radius=4,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=(
                    f"File: {r['file_name']}<br>"
                    f"Time: {r['datetime_label']}<br>"
                    f"Station: {r['station']}<br>"
                    f"Depth: {r[DEP_COL]:.2f} m<br>"
                    f"DO %: {r['DO %']:.2f}<br>"
                    f"DO mg/L: {r['DO mg/L']:.2f}<br>"
                    f"SPC: {r['SPC-uS/cm']:.2f} uS/cm<br>"
                    f"pH: {r['pH']:.2f}<br>"
                    f"ORP: {r['ORP mV']:.2f} mV<br>"
                    f"Chl: {r['Chl ug/L']:.2f}<br>"
                    f"PC: {r['PC ug/L']:.2f}<br>"
                    f"PC/Chl: {r['PC_Chl_ratio']:.3f}<br>"
                )
            ).add_to(m)

    return m


def make_summary(raw_df):
    return (
        raw_df.groupby(["
