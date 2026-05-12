import io
import zipfile
import base64
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
import folium
import streamlit as st
import streamlit.components.v1 as components

from shapely.geometry import Point


st.set_page_config(page_title="Lake Profile Analyzer", layout="wide")
st.title("Lake Water-Quality Profile Analyzer")

DATE_COL = "Date"
TIME_COL = "Time"
DEP_COL = "DEP m"
LAT_COL = "Lat"
LON_COL = "Lon"

VALUE_COLS = {
    "DO mg/L": "DO mg/L",
    "pH": "pH",
    "Chl ug/L": "Chl ug/L",
    "PC ug/L": "PC ug/L",
    "PC / Chl": "PC_Chl_ratio",
}

BASE_COLS = [
    DATE_COL, TIME_COL, DEP_COL, LAT_COL, LON_COL,
    "DO mg/L", "pH", "Chl ug/L", "PC ug/L"
]


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf


def fig_to_base64(fig):
    img_bytes = fig_to_bytes(fig).getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


def df_to_excel_bytes(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    buf.seek(0)
    return buf

# ------------------------------------------------
# BETTER PLOT STYLING
# ------------------------------------------------
def style_axis(ax, title, xlabel):
    ax.invert_yaxis()

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Depth (m)", fontsize=13)

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.7,
        alpha=0.35
    )

    ax.tick_params(labelsize=11)

    for spine in ax.spines.values():
        spine.set_alpha(0.35)


def put_legend_under(ax, ncol=3):
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=ncol,
        frameon=False,
        fontsize=10
    )

def process_file(uploaded_file):
    df = pd.read_excel(uploaded_file)

    missing = [c for c in BASE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {uploaded_file.name}: {missing}")

    df = df[BASE_COLS].copy()

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL].astype(str), format="%H:%M:%S", errors="coerce")

    for col in [DEP_COL, LAT_COL, LON_COL, "DO mg/L", "pH", "Chl ug/L", "PC ug/L"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["PC_Chl_ratio"] = df["PC ug/L"] / df["Chl ug/L"]
    df.loc[df["Chl ug/L"] == 0, "PC_Chl_ratio"] = pd.NA

    df = df.dropna(
        subset=[DATE_COL, TIME_COL, DEP_COL, LAT_COL, LON_COL]
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No valid rows found in {uploaded_file.name}")

    file_name = Path(uploaded_file.name).stem
    df["file_name"] = file_name
    df["date"] = df[DATE_COL].dt.date.astype(str)

    # ------------------------------------------------
    # 1. DEPTH-BASED LOCATION SEPARATION
    # ------------------------------------------------
    location_id = 0
    max_depth_since_zero = 0
    location_ids = []

    for dep in df[DEP_COL]:
        if dep == 0 and max_depth_since_zero > 3:
            location_id += 1
            max_depth_since_zero = 0

        location_ids.append(location_id)
        max_depth_since_zero = max(max_depth_since_zero, dep)

    df["location_id"] = location_ids

    # ------------------------------------------------
    # 2. KEEP VALID DEPTH PROFILES
    # ------------------------------------------------
    good_locations = []

    min_profile_depth_m = 3
    min_points_per_profile = 10

    for loc, g in df.groupby("location_id"):

        if g.empty or g[DEP_COL].dropna().empty:
            continue

        max_depth_group = g[DEP_COL].dropna().max()

        if (
            max_depth_group >= min_profile_depth_m
            and len(g) >= min_points_per_profile
        ):
            good_locations.append(loc)

    df = df[df["location_id"].isin(good_locations)].copy()

    if df.empty:
        raise ValueError(
            f"All locations in {uploaded_file.name} were removed as invalid profiles. "
            f"Try lowering min_profile_depth_m or min_points_per_profile."
        )

    # ------------------------------------------------
    # 2b. REMOVE MOVING GPS POINTS INSIDE EACH VALID PROFILE
    # ------------------------------------------------
    clean_parts = []

    point_radius_m = 30  # increase to 40 or 50 if valid station points are removed

    for loc, g in df.groupby("location_id"):

        if g.empty:
            continue

        center_lat = g[LAT_COL].median()
        center_lon = g[LON_COL].median()

        lat_m = (g[LAT_COL] - center_lat) * 111320
        lon_m = (g[LON_COL] - center_lon) * 111320

        dist_m = (lat_m**2 + lon_m**2) ** 0.5

        g = g.copy()
        g["dist_from_station_center_m"] = dist_m

        g_clean = g[g["dist_from_station_center_m"] <= point_radius_m].copy()

        if not g_clean.empty:
            clean_parts.append(g_clean)

    if len(clean_parts) == 0:
        raise ValueError(
            f"All points in {uploaded_file.name} were removed by point-level GPS filtering. "
            f"Try increasing point_radius_m."
        )

    df = pd.concat(clean_parts, ignore_index=True)

    # ------------------------------------------------
    # 2c. RE-CHECK EACH PROFILE AFTER POINT CLEANING
    # ------------------------------------------------
    good_locations_after_cleaning = []

    for loc, g in df.groupby("location_id"):

        if g.empty or g[DEP_COL].dropna().empty:
            continue

        if (
            g[DEP_COL].dropna().max() >= min_profile_depth_m
            and len(g) >= min_points_per_profile
        ):
            good_locations_after_cleaning.append(loc)

    df = df[df["location_id"].isin(good_locations_after_cleaning)].copy()

    if df.empty:
        raise ValueError(
            f"All locations in {uploaded_file.name} were removed after GPS point filtering. "
            f"Try increasing point_radius_m or lowering min_points_per_profile."
        )

    # ------------------------------------------------
    # 2d. RE-NUMBER LOCATIONS AFTER CLEANING
    # ------------------------------------------------
    old_to_new = {
        old: i
        for i, old in enumerate(sorted(df["location_id"].dropna().unique()))
    }

    df["location_id"] = df["location_id"].map(old_to_new)

    df = df.dropna(subset=["location_id"]).copy()
    df["location_id"] = df["location_id"].astype(int)

    df["station"] = "Station_" + (df["location_id"] + 1).astype(str)
    df["location_name"] = df["station"]

    # ------------------------------------------------
    # 3. MEAN PER DEPTH PER STATION
    # ------------------------------------------------
    mean_rows = []

    for station, g in df.groupby("station"):

        if g.empty or g[DEP_COL].dropna().empty:
            continue

        loc_id = int(g["location_id"].iloc[0])
        max_depth = int(g[DEP_COL].dropna().max())

        for meter in range(0, max_depth + 1):

            if meter == 0:
                window = g[
                    (g[DEP_COL] >= 0) &
                    (g[DEP_COL] <= 0.25)
                ]
            else:
                window = g[
                    (g[DEP_COL] >= meter - 0.25) &
                    (g[DEP_COL] <= meter + 0.25)
                ]

            if len(window) > 0:
                row = {
                    "file_name": file_name,
                    "date": g["date"].iloc[0],
                    "location_id": loc_id,
                    "station": station,
                    "location_name": station,
                    "depth_meter": meter,
                    "n_points": len(window),
                    "mean_lat": window[LAT_COL].mean(),
                    "mean_lon": window[LON_COL].mean(),
                }

                for _, value_col in VALUE_COLS.items():
                    row[f"mean_{value_col}"] = window[value_col].mean()

                mean_rows.append(row)

    # ------------------------------------------------
    # 4. LAKE MEAN
    # ------------------------------------------------
    if df.empty or df[DEP_COL].dropna().empty:
        raise ValueError(f"No valid depth data left after filtering in {uploaded_file.name}")

    max_depth = int(df[DEP_COL].dropna().max())

    for meter in range(0, max_depth + 1):

        if meter == 0:
            window = df[
                (df[DEP_COL] >= 0) &
                (df[DEP_COL] <= 0.25)
            ]
        else:
            window = df[
                (df[DEP_COL] >= meter - 0.25) &
                (df[DEP_COL] <= meter + 0.25)
            ]

        if len(window) > 0:
            row = {
                "file_name": file_name,
                "date": df["date"].iloc[0],
                "location_id": -1,
                "station": "Lake mean",
                "location_name": "Lake mean",
                "depth_meter": meter,
                "n_points": len(window),
                "mean_lat": window[LAT_COL].mean(),
                "mean_lon": window[LON_COL].mean(),
            }

            for _, value_col in VALUE_COLS.items():
                row[f"mean_{value_col}"] = window[value_col].mean()

            mean_rows.append(row)

    mean_df = pd.DataFrame(mean_rows)

    if mean_df.empty:
        raise ValueError(f"No mean values could be calculated for {uploaded_file.name}")

    return df, mean_df


# ------------------------------------------------
# LOCATION PLOT
# ------------------------------------------------
def plot_by_location(df_file, mean_file, label, col):

    mean_col = f"mean_{col}"

    fig, ax = plt.subplots(figsize=(9, 7))

    for loc, g in df_file.groupby("location_id"):

        loc_name = g["station"].iloc[0]

        # raw points
        ax.scatter(
            g[col],
            g[DEP_COL],
            s=24,
            alpha=0.22,
            linewidths=0,
            label=f"{loc_name} raw"
        )

        mg = mean_file[
            mean_file["location_id"] == loc
        ].sort_values("depth_meter")

        # mean profile
        ax.plot(
            mg[mean_col],
            mg["depth_meter"],
            marker="o",
            markersize=6,
            linewidth=2.8,
            label=f"{loc_name} mean"
        )

    style_axis(
        ax,
        title=f"{label} vs Depth - {df_file['file_name'].iloc[0]}",
        xlabel=label
    )

    put_legend_under(ax, ncol=3)

    fig.subplots_adjust(bottom=0.24)

    return fig


# ------------------------------------------------
# LAKE MEAN PLOT
# ------------------------------------------------
def plot_lake_mean(lake_mean, label, col, file_name):

    mean_col = f"mean_{col}"

    fig, ax = plt.subplots(figsize=(8, 7))

    lm = lake_mean.sort_values("depth_meter")

    ax.plot(
        lm[mean_col],
        lm["depth_meter"],
        marker="o",
        markersize=7,
        linewidth=3,
        label=file_name
    )

    style_axis(
        ax,
        title=f"Lake Mean {label} Profile - {file_name}",
        xlabel=f"Mean {label}"
    )

    put_legend_under(ax, ncol=1)

    fig.subplots_adjust(bottom=0.20)

    return fig


# ------------------------------------------------
# COMPARISON PLOT
# ------------------------------------------------
def plot_compare(compare_df, label, col):

    mean_col = f"mean_{col}"

    fig, ax = plt.subplots(figsize=(9, 7))

    for group_label, g in compare_df.groupby(["date", "file_name"]):

        date, file_name = group_label

        g = g.sort_values("depth_meter")

        ax.plot(
            g[mean_col],
            g["depth_meter"],
            marker="o",
            markersize=6,
            linewidth=2.7,
            label=f"{date}"
        )

    style_axis(
        ax,
        title=f"Comparison of Lake Mean {label}",
        xlabel=f"Mean {label}"
    )

    put_legend_under(ax, ncol=3)

    fig.subplots_adjust(bottom=0.23)

    return fig



# ------------------------------------------------
# STATIC MAP
# ------------------------------------------------
def create_static_map(all_raw_df, buffer_m=300, zoom=16):

    geometry = [
        Point(xy)
        for xy in zip(all_raw_df[LON_COL], all_raw_df[LAT_COL])
    ]

    gdf = gpd.GeoDataFrame(
        all_raw_df,
        geometry=geometry,
        crs="EPSG:4326"
    )

    gdf_web = gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(10, 10))

    for file_name, g in gdf_web.groupby("file_name"):

        ax.scatter(
            g.geometry.x,
            g.geometry.y,
            s=45,
            alpha=0.85,
            label=file_name
        )

    xmin, ymin, xmax, ymax = gdf_web.total_bounds

    ax.set_xlim(xmin - buffer_m, xmax + buffer_m)
    ax.set_ylim(ymin - buffer_m, ymax + buffer_m)

    ctx.add_basemap(
        ax,
        source=ctx.providers.Esri.WorldImagery,
        zoom=zoom
    )

    ax.set_title(
        "Sampling Points on Real Imagery",
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    ax.set_axis_off()

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=3,
        frameon=False,
        fontsize=10
    )

    fig.subplots_adjust(bottom=0.10)

    return fig


def create_interactive_map(all_raw_df, zoom_start=15):
    center_lat = all_raw_df[LAT_COL].mean()
    center_lon = all_raw_df[LON_COL].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="Esri.WorldImagery"
    )

    colors = ["blue", "red", "green", "purple", "orange", "darkred", "cadetblue", "black", "pink"]

    for i, (file_name, g) in enumerate(all_raw_df.groupby("file_name")):
        color = colors[i % len(colors)]

        for _, row in g.iterrows():
            folium.CircleMarker(
                location=[row[LAT_COL], row[LON_COL]],
                radius=4,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=(
                    f"File: {file_name}<br>"
                    f"Date: {row['date']}<br>"
                    f"Location: {row['location_name']}<br>"
                    f"Depth: {row[DEP_COL]:.2f} m<br>"
                    f"DO: {row['DO mg/L']:.2f} mg/L<br>"
                    f"pH: {row['pH']:.2f}<br>"
                    f"Chl: {row['Chl ug/L']:.2f} ug/L<br>"
                    f"PC: {row['PC ug/L']:.2f} ug/L<br>"
                    f"PC/Chl: {row['PC_Chl_ratio']:.3f}"
                )
            ).add_to(m)

    return m


def create_html_report(all_raw_df, all_mean_df, selected_label, selected_col, fig_compare, fig_map=None):
    summary_table = (
        all_raw_df
        .groupby(["file_name", "date"])
        .agg(
            n_points=(DEP_COL, "count"),
            max_depth_m=(DEP_COL, "max"),
            mean_DO=("DO mg/L", "mean"),
            mean_pH=("pH", "mean"),
            mean_Chl=("Chl ug/L", "mean"),
            mean_PC=("PC ug/L", "mean"),
            mean_PC_Chl=("PC_Chl_ratio", "mean"),
        )
        .reset_index()
        .round(3)
    )

    compare_img = fig_to_base64(fig_compare)

    map_section = ""
    if fig_map is not None:
        map_img = fig_to_base64(fig_map)
        map_section = f"""
        <h2>Static Sampling Map</h2>
        <img src="data:image/png;base64,{map_img}" style="max-width:900px;width:100%;">
        """

    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>Lake Water-Quality Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2 {{ color: #1f4e79; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 25px; }}
            th, td {{ border: 1px solid #ccc; padding: 6px; text-align: left; }}
            th {{ background: #f0f0f0; }}
            img {{ margin: 15px 0; }}
        </style>
    </head>
    <body>
        <h1>Lake Water-Quality Profile Report</h1>

        <h2>Summary</h2>
        {summary_table.to_html(index=False)}

        <h2>Selected Variable</h2>
        <p>{selected_label}</p>

        <h2>Comparison Between Dates / Files</h2>
        <img src="data:image/png;base64,{compare_img}" style="max-width:800px;width:100%;">

        {map_section}

        <h2>Mean Table</h2>
        {all_mean_df.round(3).to_html(index=False)}
    </body>
    </html>
    """

    return html


def create_all_variables_report(all_raw_df, all_mean_df, fig_map=None):
    sections = ""

    for var_label, var_col in VALUE_COLS.items():
        compare_df = all_mean_df[all_mean_df["location_name"] == "Lake mean"]
        fig_compare = plot_compare(compare_df, var_label, var_col)
        compare_img = fig_to_base64(fig_compare)
        plt.close(fig_compare)

        sections += f"""
        <h2>{var_label}</h2>
        <h3>Comparison Between Dates / Files</h3>
        <img src="data:image/png;base64,{compare_img}" style="max-width:850px;width:100%;">
        """

        for file_name, df_file in all_raw_df.groupby("file_name"):
            mean_file = all_mean_df[
                (all_mean_df["file_name"] == file_name) &
                (all_mean_df["location_name"] != "Lake mean")
            ]

            lake_mean = all_mean_df[
                (all_mean_df["file_name"] == file_name) &
                (all_mean_df["location_name"] == "Lake mean")
            ]

            fig_loc = plot_by_location(df_file, mean_file, var_label, var_col)
            loc_img = fig_to_base64(fig_loc)
            plt.close(fig_loc)

            fig_lake = plot_lake_mean(lake_mean, var_label, var_col, file_name)
            lake_img = fig_to_base64(fig_lake)
            plt.close(fig_lake)

            sections += f"""
            <h3>{file_name}</h3>
            <h4>Split by Location</h4>
            <img src="data:image/png;base64,{loc_img}" style="max-width:850px;width:100%;">

            <h4>Lake Mean</h4>
            <img src="data:image/png;base64,{lake_img}" style="max-width:850px;width:100%;">
            """

    map_section = ""
    if fig_map is not None:
        map_img = fig_to_base64(fig_map)
        map_section = f"""
        <h2>Static Sampling Map</h2>
        <img src="data:image/png;base64,{map_img}" style="max-width:900px;width:100%;">
        """

    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>Full Lake Water-Quality Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #12395a; }}
            h2 {{ color: #1f4e79; border-bottom: 2px solid #ddd; padding-bottom: 6px; }}
            h3 {{ color: #333; margin-top: 25px; }}
            h4 {{ color: #555; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 25px; }}
            th, td {{ border: 1px solid #ccc; padding: 6px; text-align: left; }}
            th {{ background: #f0f0f0; }}
            img {{ margin: 15px 0 30px 0; }}
        </style>
    </head>
    <body>
        <h1>Full Lake Water-Quality Profile Report</h1>

        {map_section}

        {sections}

        <h2>All Mean Values</h2>
        {all_mean_df.round(3).to_html(index=False)}
    </body>
    </html>
    """

    return html
    
uploaded_files = st.sidebar.file_uploader(
    "Drag Excel files here",
    type=["xlsx", "xls"],
    accept_multiple_files=True
)

st.sidebar.header("Settings")
buffer_m = st.sidebar.slider("Static map zoom-out buffer, meters", 50, 1500, 300, 50)
static_zoom = st.sidebar.slider("Static map imagery zoom", 12, 20, 16, 1)
interactive_zoom = st.sidebar.slider("Interactive map zoom", 10, 20, 15, 1)

selected_variable = st.sidebar.selectbox("Variable to display", list(VALUE_COLS.keys()))

if not uploaded_files:
    st.info("Upload one or more Excel files to start.")
    st.stop()

all_raw = []
all_means = []
errors = []

for uploaded_file in uploaded_files:
    try:
        raw_df, mean_df = process_file(uploaded_file)
        all_raw.append(raw_df)
        all_means.append(mean_df)
    except Exception as e:
        errors.append(str(e))

if errors:
    st.error("\n".join(errors))
    st.stop()

all_raw_df = pd.concat(all_raw, ignore_index=True)
all_mean_df = pd.concat(all_means, ignore_index=True)

def assign_global_station_names(all_raw_df, all_mean_df, match_radius_m=40):
    station_centers = (
        all_raw_df
        .groupby(["file_name", "date", "location_id"])
        .agg(
            mean_lat=(LAT_COL, "mean"),
            mean_lon=(LON_COL, "mean")
        )
        .reset_index()
    )

    global_stations = []
    station_names = {}

    next_id = 1

    for _, row in station_centers.iterrows():
        assigned = None

        for s in global_stations:
            dist_m = (
                ((row["mean_lat"] - s["lat"]) * 111320) ** 2 +
                ((row["mean_lon"] - s["lon"]) * 111320) ** 2
            ) ** 0.5

            if dist_m <= match_radius_m:
                assigned = s["name"]
                break

        if assigned is None:
            assigned = f"Station_{next_id}"
            global_stations.append({
                "name": assigned,
                "lat": row["mean_lat"],
                "lon": row["mean_lon"],
            })
            next_id += 1

        station_names[(row["file_name"], row["location_id"])] = assigned

    all_raw_df["station"] = all_raw_df.apply(
        lambda r: station_names.get((r["file_name"], r["location_id"]), "Unknown"),
        axis=1
    )
    all_raw_df["location_name"] = all_raw_df["station"]

    all_mean_df["station"] = all_mean_df.apply(
        lambda r: "Lake mean" if r["location_id"] == -1 else station_names.get((r["file_name"], r["location_id"]), "Unknown"),
        axis=1
    )
    all_mean_df["location_name"] = all_mean_df["station"]

    return all_raw_df, all_mean_df

all_raw_df, all_mean_df = assign_global_station_names(
    all_raw_df,
    all_mean_df,
    match_radius_m=40
)

label = selected_variable
col = VALUE_COLS[label]

# ------------------------------------------------
# TABLES
# ------------------------------------------------
st.header("Tables")

st.subheader("Summary by Station")

summary_df = (
    all_raw_df
    .groupby(["date", "file_name", "station"])
    .agg(
        n_points=(DEP_COL, "count"),
        max_depth_m=(DEP_COL, "max"),
        mean_DO_mg_L=("DO mg/L", "mean"),
        mean_pH=("pH", "mean"),
        mean_Chl_ug_L=("Chl ug/L", "mean"),
        mean_PC_ug_L=("PC ug/L", "mean"),
        mean_PC_Chl_ratio=("PC_Chl_ratio", "mean"),
        mean_lat=(LAT_COL, "mean"),
        mean_lon=(LON_COL, "mean"),
    )
    .reset_index()
    .round(3)
)

st.dataframe(summary_df, use_container_width=True)

st.download_button(
    "Download summary by station CSV",
    summary_df.to_csv(index=False).encode("utf-8"),
    file_name="summary_by_station.csv",
    mime="text/csv"
)

c1, c2, c3 = st.columns(3)

with c1:
    st.download_button(
        "Download raw data CSV",
        all_raw_df.to_csv(index=False).encode("utf-8"),
        file_name="all_raw_data.csv",
        mime="text/csv"
    )

with c2:
    st.download_button(
        "Download means CSV",
        all_mean_df.to_csv(index=False).encode("utf-8"),
        file_name="all_means.csv",
        mime="text/csv"
    )

with c3:
    st.download_button(
        "Download means Excel",
        df_to_excel_bytes(all_mean_df),
        file_name="all_means.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.dataframe(all_mean_df.round(3), use_container_width=True)


# ------------------------------------------------
# PROFILES SPLIT BY LOCATION
# ------------------------------------------------
st.header("Profiles Split by Location")

for file_name, df_file in all_raw_df.groupby("file_name"):

    st.subheader(file_name)

    mean_file = all_mean_df[
        (all_mean_df["file_name"] == file_name) &
        (all_mean_df["location_name"] != "Lake mean")
    ]

    lake_mean = all_mean_df[
        (all_mean_df["file_name"] == file_name) &
        (all_mean_df["location_name"] == "Lake mean")
    ]

    fig1 = plot_by_location(df_file, mean_file, label, col)
    st.pyplot(fig1)

    st.download_button(
        f"Download {file_name} location plot PNG",
        fig_to_bytes(fig1),
        file_name=f"{file_name}_{col}_by_location.png",
        mime="image/png"
    )

    fig2 = plot_lake_mean(lake_mean, label, col, file_name)
    st.pyplot(fig2)

    st.download_button(
        f"Download {file_name} lake mean plot PNG",
        fig_to_bytes(fig2),
        file_name=f"{file_name}_{col}_lake_mean.png",
        mime="image/png"
    )

    st.dataframe(
        all_mean_df[all_mean_df["file_name"] == file_name].round(3),
        use_container_width=True
    )


# ------------------------------------------------
# COMPARISON BETWEEN DATES / FILES
# ------------------------------------------------
st.header("Comparison Between Dates / Files")

compare_df = all_mean_df[all_mean_df["location_name"] == "Lake mean"]

fig_compare = plot_compare(compare_df, label, col)
st.pyplot(fig_compare)

st.download_button(
    "Download comparison plot PNG",
    fig_to_bytes(fig_compare),
    file_name=f"comparison_lake_mean_{col}.png",
    mime="image/png"
)


# ------------------------------------------------
# MAPS
# ------------------------------------------------
st.header("Maps")

st.subheader("Interactive Map")

m = create_interactive_map(all_raw_df, zoom_start=interactive_zoom)
map_html = m.get_root().render()

components.html(map_html, height=650, scrolling=True)

st.download_button(
    "Download interactive map HTML",
    map_html.encode("utf-8"),
    file_name="interactive_sampling_map.html",
    mime="text/html"
)

st.subheader("Static Imagery Map")

fig_map = None

try:
    fig_map = create_static_map(
        all_raw_df,
        buffer_m=buffer_m,
        zoom=static_zoom
    )

    st.pyplot(fig_map)

    st.download_button(
        "Download static map PNG",
        fig_to_bytes(fig_map),
        file_name="static_sampling_map.png",
        mime="image/png"
    )

except Exception as e:
    st.warning(f"Static map failed. Interactive map still works. Error: {e}")


# ------------------------------------------------
# REPORTS
# ------------------------------------------------
st.header("Reports")

# selected-variable report
report_html = create_html_report(
    all_raw_df=all_raw_df,
    all_mean_df=all_mean_df,
    selected_label=label,
    selected_col=col,
    fig_compare=fig_compare,
    fig_map=fig_map
)

st.subheader("Selected Variable Report")

components.html(report_html, height=800, scrolling=True)

st.download_button(
    "Download selected-variable HTML report",
    report_html.encode("utf-8"),
    file_name=f"lake_water_quality_report_{col}.html",
    mime="text/html"
)

# full all-variables report
full_report_html = create_all_variables_report(
    all_raw_df=all_raw_df,
    all_mean_df=all_mean_df,
    fig_map=fig_map
)

st.subheader("Full Report - All Variables")

st.download_button(
    "Download FULL report - all variables",
    full_report_html.encode("utf-8"),
    file_name="full_lake_water_quality_report_all_variables.html",
    mime="text/html"
)


# ------------------------------------------------
# DOWNLOAD ALL OUTPUTS AS ZIP
# ------------------------------------------------
st.header("Download All Outputs")

zip_buffer = io.BytesIO()

with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:

    # tables
    z.writestr("tables/all_raw_data.csv", all_raw_df.to_csv(index=False))
    z.writestr("tables/all_means.csv", all_mean_df.to_csv(index=False))

    # reports
    z.writestr(
        f"reports/lake_water_quality_report_{col}.html",
        report_html
    )

    z.writestr(
        "reports/full_lake_water_quality_report_all_variables.html",
        full_report_html
    )

    # maps
    z.writestr("maps/interactive_sampling_map.html", map_html)

    if fig_map is not None:
        z.writestr(
            "maps/static_sampling_map.png",
            fig_to_bytes(fig_map).getvalue()
        )

    # plots for every variable and every file
    for file_name, df_file in all_raw_df.groupby("file_name"):

        mean_file = all_mean_df[
            (all_mean_df["file_name"] == file_name) &
            (all_mean_df["location_name"] != "Lake mean")
        ]

        lake_mean = all_mean_df[
            (all_mean_df["file_name"] == file_name) &
            (all_mean_df["location_name"] == "Lake mean")
        ]

        for var_label, var_col in VALUE_COLS.items():

            fig1 = plot_by_location(
                df_file,
                mean_file,
                var_label,
                var_col
            )

            z.writestr(
                f"plots/{file_name}_{var_col}_by_location.png",
                fig_to_bytes(fig1).getvalue()
            )

            plt.close(fig1)

            fig2 = plot_lake_mean(
                lake_mean,
                var_label,
                var_col,
                file_name
            )

            z.writestr(
                f"plots/{file_name}_{var_col}_lake_mean.png",
                fig_to_bytes(fig2).getvalue()
            )

            plt.close(fig2)

    # comparison plots for every variable
    for var_label, var_col in VALUE_COLS.items():

        fig = plot_compare(
            compare_df,
            var_label,
            var_col
        )

        z.writestr(
            f"plots/comparison_lake_mean_{var_col}.png",
            fig_to_bytes(fig).getvalue()
        )

        plt.close(fig)

zip_buffer.seek(0)

st.download_button(
    "Download ZIP with tables, plots, maps, and reports",
    zip_buffer,
    file_name="lake_profile_outputs.zip",
    mime="application/zip"
)
