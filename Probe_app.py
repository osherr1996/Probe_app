import io
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import folium
import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="Lake Profile Analyzer", layout="wide")
st.title("Lake Water-Quality Profile Analyzer")

DATE_COL = "Date"
TIME_COL = "Time"
DEP_COL = "DEP m"
LAT_COL = "Lat"
LON_COL = "Lon"

VALUE_COLS = {
    "DO %": "DO %",
    "DO mg/L": "DO mg/L",
    "pH": "pH",
    "ORP mV": "ORP mV",
    "Chl ug/L": "Chl ug/L",
    "PC ug/L": "PC ug/L",
    "PC / Chl": "PC_Chl_ratio",
}

BASE_COLS = [
    DATE_COL, TIME_COL, DEP_COL, LAT_COL, LON_COL,
    "DO %", "DO mg/L", "pH", "ORP mV", "Chl ug/L", "PC ug/L"
]


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
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

    for col in [DEP_COL, LAT_COL, LON_COL, "DO %", "DO mg/L", "pH", "ORP mV", "Chl ug/L", "PC ug/L"]:
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
    
    df["time_range"] = (
        df["start_time"] + " - " + df["end_time"]
    )
    
    df["datetime_label"] = (
        df[DATE_COL].dt.strftime("%d/%m/%Y") +
        "  " +
        df["time_range"]
    )

    # depth-based profile separation
    location_id = 0
    max_depth_since_zero = 0
    ids = []

    for dep in df[DEP_COL]:
        if dep == 0 and max_depth_since_zero > 3:
            location_id += 1
            max_depth_since_zero = 0

        ids.append(location_id)
        max_depth_since_zero = max(max_depth_since_zero, dep)

    df["location_id"] = ids

    # keep valid profiles
    good = []
    for loc, g in df.groupby("location_id"):
        if len(g) >= 10 and g[DEP_COL].max() >= 3:
            good.append(loc)

    df = df[df["location_id"].isin(good)].copy()

    if df.empty:
        raise ValueError(f"No valid depth profiles found in {uploaded_file.name}")

    # remove moving GPS points inside each profile
    parts = []
    point_radius_m = 30

    for loc, g in df.groupby("location_id"):
        center_lat = g[LAT_COL].median()
        center_lon = g[LON_COL].median()

        lat_m = (g[LAT_COL] - center_lat) * 111320
        lon_m = (g[LON_COL] - center_lon) * 111320
        dist_m = (lat_m ** 2 + lon_m ** 2) ** 0.5

        g = g.copy()
        g["dist_from_station_center_m"] = dist_m
        g = g[g["dist_from_station_center_m"] <= point_radius_m]

        if len(g) >= 10 and g[DEP_COL].max() >= 3:
            parts.append(g)

    if not parts:
        raise ValueError(f"All points removed by GPS filtering in {uploaded_file.name}")

    df = pd.concat(parts, ignore_index=True)

    # re-number stations inside file
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

    # lake mean
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


def assign_global_station_names(raw_df, mean_df, radius_m=40):
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
            d = (((r["mean_lat"] - s["lat"]) * 111320) ** 2 +
                 ((r["mean_lon"] - s["lon"]) * 111320) ** 2) ** 0.5

            if d <= radius_m:
                assigned = s["name"]
                break

        if assigned is None:
            assigned = f"Station_{next_id}"
            global_stations.append({"name": assigned, "lat": r["mean_lat"], "lon": r["mean_lon"]})
            next_id += 1

        names[(r["file_name"], r["location_id"])] = assigned

    raw_df["station"] = raw_df.apply(lambda r: names[(r["file_name"], r["location_id"])], axis=1)
    raw_df["location_name"] = raw_df["station"]

    mean_df["station"] = mean_df.apply(
        lambda r: "Lake mean" if r["location_id"] == -1 else names.get((r["file_name"], r["location_id"]), "Unknown"),
        axis=1
    )
    mean_df["location_name"] = mean_df["station"]

    return raw_df, mean_df


def style_profile(ax, title, xlabel):
    ax.invert_yaxis()  # 0 depth at top
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Depth (m)")

    ymin, ymax = ax.get_ylim()
    max_depth = int(max(ymin, ymax))
    ax.set_yticks(range(0, max_depth + 1))

    ax.grid(True, linestyle="--", alpha=0.35)
    
def plot_all_variables_for_file(raw_file, mean_file, file_name):
    fig, axes = plt.subplots(1, len(VALUE_COLS), figsize=(26, 5), sharey=False)

    for ax, (label, col) in zip(axes, VALUE_COLS.items()):
        mean_col = f"mean_{col}"

        for station, g in raw_file.groupby("station"):
            ax.scatter(g[col], g[DEP_COL], s=14, alpha=0.20)

            mg = mean_file[mean_file["station"] == station].sort_values("depth_meter")
            ax.plot(mg[mean_col], mg["depth_meter"], marker="o", linewidth=2, label=station)

        style_profile(ax, label, label)

    axes[0].legend(loc="upper center", bbox_to_anchor=(3.2, -0.18), ncol=5, frameon=False)
    time_label = raw_file["datetime_label"].iloc[0]
    fig.suptitle( f"{file_name}\n{time_label}", fontsize=16,  fontweight="bold", y=0.98)
    fig.subplots_adjust( bottom=0.25, top=0.72, wspace=0.35)

    return fig


def plot_comparison(mean_df):
    lake = mean_df[mean_df["station"] == "Lake mean"]

    fig, axes = plt.subplots(
        1,
        len(VALUE_COLS),
        figsize=(30, 5),
        sharey=False
    )

    for ax, (label, col) in zip(axes, VALUE_COLS.items()):

        mean_col = f"mean_{col}"

        for key, g in lake.groupby(["date", "file_name"]):

            date, file_name = key

            g = g.sort_values("depth_meter")

            if "time_range" in g.columns:
                time_range = g["time_range"].iloc[0]
            else:
                time_range = ""

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

    fig.suptitle(
        "Lake Mean Comparison Between Data",
        fontsize=16,
        fontweight="bold"
    )

    fig.subplots_adjust(
        bottom=0.25,
        top=0.85
    )

    return fig


def create_map(df_file, zoom=15):
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
                    f"Date: {r['date']}<br>"
                    f"Station: {r['station']}<br>"
                    f"Depth: {r[DEP_COL]:.2f} m<br>"
                    f"DO: {r['DO mg/L']:.2f}<br>"
                    f"pH: {r['pH']:.2f}<br>"
                    f"Chl: {r['Chl ug/L']:.2f}<br>"
                    f"PC: {r['PC ug/L']:.2f}<br>"
                    f"PC/Chl: {r['PC_Chl_ratio']:.3f}"
                    f"Time: {r['datetime_label']}<br>"
                    f"ORP: {r['ORP mV']:.2f} mV<br>"
                )
            ).add_to(m)

    return m

def make_summary(raw_df):
    return (
        raw_df.groupby(["date", "file_name", "station"])
        .agg(
            time_range=("time_range", "first"),
            datetime_label=("datetime_label", "first"),
            n_points=(DEP_COL, "count"),
            max_depth_m=(DEP_COL, "max"),
            mean_DO_percent=("DO %", "mean"),
            mean_DO_mg_L=("DO mg/L", "mean"),
            mean_pH=("pH", "mean"),
            mean_ORP_mV=("ORP mV", "mean"),
            mean_Chl_ug_L=("Chl ug/L", "mean"),
            mean_PC_ug_L=("PC ug/L", "mean"),
            mean_PC_Chl_ratio=("PC_Chl_ratio", "mean"),
            mean_lat=(LAT_COL, "mean"),
            mean_lon=(LON_COL, "mean"),
        )
        .reset_index()
        .round(3)
    )



uploaded_files = st.sidebar.file_uploader(
    "Upload Excel files",
    type=["xlsx", "xls"],
    accept_multiple_files=True
)

map_zoom = st.sidebar.slider("Map zoom", 10, 20, 15)
match_radius = st.sidebar.slider(
    "Match station between data (meters)",
    10,
    100,
    40
)

if not uploaded_files:
    st.info("Upload Excel files to start.")
    st.stop()

raw_parts = []
errors = []

for f in uploaded_files:
    try:
        raw_parts.append(process_file(f))
    except Exception as e:
        errors.append(f"{f.name}: {e}")

if errors:
    st.error("\n".join(errors))
    st.stop()

# ------------------------------------------------
# MERGE ALL FILES
# ------------------------------------------------
raw_df = pd.concat(raw_parts, ignore_index=True)

# initial means
mean_df = calculate_means(raw_df)

# assign same station names between dates/files
raw_df, mean_df = assign_global_station_names(
    raw_df,
    mean_df,
    radius_m=match_radius
)

# update station names in mean_df
mean_df["station"] = mean_df.apply(
    lambda r: (
        "Lake mean"
        if r["location_id"] == -1
        else raw_df[
            (raw_df["file_name"] == r["file_name"]) &
            (raw_df["location_id"] == r["location_id"])
        ]["station"].iloc[0]
    ),
    axis=1
)

mean_df["location_name"] = mean_df["station"]

# ------------------------------------------------
# SUMMARY
# ------------------------------------------------
summary_df = make_summary(raw_df)

# ------------------------------------------------
# FILE SELECTION
# ------------------------------------------------
file_options = sorted(raw_df["file_name"].unique())

selected_file = st.sidebar.selectbox(
    "Choose data",
    file_options
)

df_file = raw_df[
    raw_df["file_name"] == selected_file
]

mean_file = mean_df[
    (mean_df["file_name"] == selected_file) &
    (mean_df["station"] != "Lake mean")
]

# ------------------------------------------------
# MAIN DATA PANEL
# ------------------------------------------------
st.header(f"Data: {selected_file}")

time_label = df_file["datetime_label"].iloc[0]

st.markdown(
    f"### Sampling Time: `{time_label}`"
)

fig_data = plot_all_variables_for_file(
    df_file,
    mean_file,
    selected_file
)

st.pyplot(fig_data)

st.download_button(
    "Download all-variable station plot",
    fig_to_bytes(fig_data),
    file_name=f"{clean_filename(selected_file)}_all_variables_by_station.png",
    mime="image/png"
)

# ------------------------------------------------
# MAP
# ------------------------------------------------
st.subheader("Map by Station")

m = create_map(df_file, zoom=map_zoom)

map_html = m.get_root().render()

components.html(
    map_html,
    height=600,
    scrolling=True
)

st.download_button(
    "Download interactive map",
    map_html.encode("utf-8"),
    file_name=f"{clean_filename(selected_file)}_map.html",
    mime="text/html"
)

# ------------------------------------------------
# COMPARISON PANEL
# ------------------------------------------------
st.header("Comparison Panel")

fig_compare = plot_comparison(mean_df)

st.pyplot(fig_compare)

st.download_button(
    "Download comparison plot",
    fig_to_bytes(fig_compare),
    file_name="lake_mean_comparison_all_variables.png",
    mime="image/png"
)

# ------------------------------------------------
# DOWNLOAD SUMMARY
# ------------------------------------------------
st.download_button(
    "Download summary CSV",
    summary_df.to_csv(index=False).encode("utf-8"),
    file_name="summary_by_station_between_data.csv",
    mime="text/csv"
)

# ------------------------------------------------
# DOWNLOAD EACH VARIABLE COMPARISON
# ------------------------------------------------
st.subheader("Download Mean Comparison per Variable")

for var_label, var_col in VALUE_COLS.items():

    lake = mean_df[
        mean_df["station"] == "Lake mean"
    ]

    mean_col = f"mean_{var_col}"

    fig_one, ax = plt.subplots(figsize=(7, 6))

    for key, g in lake.groupby(["date", "file_name"]):

        date, file_name = key

        g = g.sort_values("depth_meter")

        if "time_period" in g.columns:
            period = g["time_period"].iloc[0]
        else:
            period = ""

        ax.plot(
            g[mean_col],
            g["depth_meter"],
            marker="o",
            linewidth=2,
            label=f"{date} ({period})"
        )

    style_profile(
        ax,
        var_label,
        f"Mean {var_label}"
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=False
    )

    fig_one.subplots_adjust(bottom=0.22)

    st.download_button(
        f"Download mean comparison plot - {var_label}",
        fig_to_bytes(fig_one),
        file_name=f"mean_comparison_{clean_filename(var_label)}.png",
        mime="image/png"
    )

# ------------------------------------------------
# SUMMARY PREVIEW
# ------------------------------------------------
with st.expander("Preview summary"):
    st.dataframe(
        summary_df,
        use_container_width=True
    )
