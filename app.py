from flask import Flask, request, jsonify, render_template
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px

# =========================
# Flask server
# =========================
server = Flask(__name__)

# =========================
# Load dataset
# =========================
df = pd.read_csv("crash_person_merged.csv", index_col=0, low_memory=False)

# Ensure CRASH_YEAR exists
if "CRASH DATE" in df.columns and "CRASH_YEAR" not in df.columns:
    df["CRASH DATE"] = pd.to_datetime(df["CRASH DATE"], errors="coerce")
    df["CRASH_YEAR"] = df["CRASH DATE"].dt.year

# Prepare dropdown options
borough_options = sorted(df["BOROUGH"].dropna().unique()) if "BOROUGH" in df.columns else []
year_options = sorted(df["CRASH_YEAR"].dropna().unique()) if "CRASH_YEAR" in df.columns else []

vehicle_cols = [c for c in df.columns if "VEHICLE TYPE CODE" in c]
vehicle_type_options = []
for col in vehicle_cols:
    vehicle_type_options.extend(df[col].dropna().unique())
vehicle_type_options = sorted(list(set(vehicle_type_options)))

factor_cols = [c for c in df.columns if "CONTRIBUTING FACTOR VEHICLE" in c]
contributing_factor_options = []
for col in factor_cols:
    contributing_factor_options.extend(df[col].dropna().unique())
contributing_factor_options = sorted(list(set(contributing_factor_options)))

injury_type_options = [
    {"label": "All", "value": "all"},
    {"label": "Injured", "value": "injured"},
    {"label": "Killed", "value": "killed"},
]

# =========================
# Dash app
# =========================
app = Dash(__name__, server=server, url_base_pathname="/dash/")

# Colors
BACKGROUND_COLOR = "#0f172a"
TEXT_COLOR = "#e5e7eb"
CARD_BG = "#1e293b"
ACCENT = "#3b82f6"
ACCENT_SOFT = "#475569"
MUTED_TEXT = "#94a3b8"

app.layout = html.Div(
    style={
        "margin": "0",
        "padding": "0",
        "minHeight": "100vh",
        "backgroundColor": BACKGROUND_COLOR,
        "color": TEXT_COLOR,
        "fontFamily": "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        "display": "flex",
        "flexDirection": "column",
    },
    children=[
        html.H1("NYC Crash Dashboard", style={"textAlign": "center", "margin": "16px 0"}),

        html.Div(
            style={"display": "flex", "gap": "20px", "padding": "0 20px", "flexWrap": "wrap"},
            children=[
                html.Div(
                    style={"flex": "0 0 300px", "display": "flex", "flexDirection": "column", "gap": "10px"},
                    children=[
                        html.Label("Borough"),
                        dcc.Dropdown(
                            id="borough-dropdown",
                            options=[{"label": b, "value": b} for b in borough_options],
                            placeholder="All boroughs",
                        ),
                        html.Label("Year"),
                        dcc.Dropdown(
                            id="year-dropdown",
                            options=[{"label": str(y), "value": y} for y in year_options],
                            placeholder="All years",
                        ),
                        html.Label("Injury Type"),
                        dcc.Dropdown(
                            id="injury-type-dropdown",
                            options=injury_type_options,
                            value="all",
                            clearable=False,
                        ),
                        html.Label("Vehicle Type"),
                        dcc.Dropdown(
                            id="vehicle-type-dropdown",
                            options=[{"label": v, "value": v} for v in vehicle_type_options],
                            multi=True,
                        ),
                        html.Label("Contributing Factor"),
                        dcc.Dropdown(
                            id="contrib-factor-dropdown",
                            options=[{"label": f, "value": f} for f in contributing_factor_options],
                            multi=True,
                        ),
                        html.Button("Generate Report", id="generate-button", n_clicks=0),
                    ],
                ),
                html.Div(
                    style={"flex": "1 1 0", "display": "flex", "flexDirection": "column", "gap": "12px"},
                    children=[
                        html.Div(id="summary-div"),
                        dcc.Graph(id="crashes-by-borough", style={"height": "400px"}),
                        dcc.Graph(id="time-series-crashes", style={"height": "400px"}),
                        dcc.Graph(id="hour-dow-heatmap", style={"height": "400px"}),
                        dcc.Graph(id="injury-pie", style={"height": "400px"}),
                        dcc.Graph(id="map-crashes", style={"height": "500px"}),
                    ],
                ),
            ],
        ),
    ],
)

@app.callback(
    [
        Output("crashes-by-borough", "figure"),
        Output("time-series-crashes", "figure"),
        Output("hour-dow-heatmap", "figure"),
        Output("injury-pie", "figure"),
        Output("map-crashes", "figure"),
        Output("summary-div", "children"),
    ],
    [Input("generate-button", "n_clicks")],
    [
        State("borough-dropdown", "value"),
        State("year-dropdown", "value"),
        State("injury-type-dropdown", "value"),
        State("vehicle-type-dropdown", "value"),
        State("contrib-factor-dropdown", "value"),
    ],
)
def update_report(n_clicks, borough_value, year_value, injury_type_value, vehicle_types, contrib_factors):
    filtered = df.copy()

    if borough_value:
        filtered = filtered[filtered["BOROUGH"] == borough_value]
    if year_value:
        filtered = filtered[filtered["CRASH_YEAR"] == year_value]
    if injury_type_value == "injured":
        filtered = filtered[filtered["NUMBER OF PERSONS INJURED"] > 0]
    if injury_type_value == "killed":
        filtered = filtered[filtered["NUMBER OF PERSONS KILLED"] > 0]
    if vehicle_types:
        vehicle_mask = pd.Series(False, index=filtered.index)
        for c in vehicle_cols:
            if c in filtered.columns:
                vehicle_mask |= filtered[c].isin(vehicle_types)
        filtered = filtered[vehicle_mask]
    if contrib_factors:
        factor_mask = pd.Series(False, index=filtered.index)
        for c in factor_cols:
            if c in filtered.columns:
                factor_mask |= filtered[c].isin(contrib_factors)
        filtered = filtered[factor_mask]

    empty_fig = {"data": [], "layout": {"xaxis": {"visible": False}, "yaxis": {"visible": False}, "annotations": []}}
    if filtered.empty:
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, "No crashes match the filters."

    bar_df = filtered.groupby("BOROUGH", dropna=False).size().reset_index(name="count")
    bar_fig = px.bar(bar_df, x="BOROUGH", y="count", title="Crashes by Borough", template="plotly_dark")

    ts_df = filtered.groupby("CRASH_YEAR").size().reset_index(name="count")
    ts_fig = px.line(ts_df, x="CRASH_YEAR", y="count", markers=True, title="Crashes over Years", template="plotly_dark")

    if "HOUR" in filtered.columns and "DAY_OF_WEEK" in filtered.columns:
        heat_df = filtered.groupby(["DAY_OF_WEEK", "HOUR"]).size().reset_index(name="count")
        heat_fig = px.density_heatmap(heat_df, x="HOUR", y="DAY_OF_WEEK", z="count", template="plotly_dark", title="Crash Density by Hour/DOW")
    else:
        heat_fig = empty_fig

    injured = filtered["NUMBER OF PERSONS INJURED"].sum()
    killed = filtered["NUMBER OF PERSONS KILLED"].sum()
    pie_df = pd.DataFrame({"category": ["Injured", "Killed"], "count": [injured, killed]})
    pie_fig = px.pie(pie_df, names="category", values="count", title="Injured vs Killed", hole=0.4, template="plotly_dark")

    if "LATITUDE" in filtered.columns and "LONGITUDE" in filtered.columns:
        map_df = filtered.dropna(subset=["LATITUDE", "LONGITUDE"])
        if len(map_df) > 10000:
            map_df = map_df.sample(10000, random_state=42)
        map_fig = px.scatter_mapbox(map_df, lat="LATITUDE", lon="LONGITUDE", hover_name="BOROUGH",
                                    hover_data={"CRASH DATE": True}, zoom=9, height=500, title="Crash Locations (Sampled)")
        map_fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":30,"l":0,"b":0})
    else:
        map_fig = empty_fig

    summary_text = f"Showing {len(filtered)} crashes after filters."
    return bar_fig, ts_fig, heat_fig, pie_fig, map_fig, summary_text

if __name__ == "__main__":
    app.run(debug=True)