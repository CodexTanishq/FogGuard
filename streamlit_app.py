import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

# ---------------------------------------------------------------------------
# Page Config & Custom CSS (appV3 style)
# ---------------------------------------------------------------------------
st.set_page_config(page_title="FogGuard | Central Control", page_icon="🛡️", layout="wide")

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    /* Override Streamlit Theme */
    .stApp {
        background-color: #0a0d10;
        color: #eae7e0;
        font-family: 'IBM Plex Mono', monospace;
    }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }
    
    /* Variables from V3 */
    :root {
        --bg-void: #0a0d10; --bg-panel: #12161b; --bg-raised: #171c22;
        --line: #262c33; --line-bright: #3a4149;
        --text-hi: #eae7e0; --text-muted: #838d97;
        --amber: #ffb020; --cyan: #29d3c7;
        --c-crit: #ff4d4f; --c-adv: #f5a623; --c-clear: #35d07f;
    }

    h1, h2, h3, h4, h5, h6 { font-family: 'Rajdhani', sans-serif; font-weight: 600; letter-spacing: 0.01em; color: var(--text-hi); }
    
    /* V3 UI Components */
    .fg-mast { display: flex; align-items: center; justify-content: space-between; padding: 0.9rem 1.3rem;
        margin-bottom: 1.4rem; background: linear-gradient(180deg, var(--bg-raised), var(--bg-panel));
        border: 1px solid var(--line); border-left: 3px solid var(--amber); border-radius: 3px; }
    .fg-mast-title { font-family: 'Rajdhani', sans-serif; font-size: 1.5rem; font-weight: 700; margin: 0; }
    .fg-mast-sub { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: var(--text-muted); letter-spacing: 0.08em; text-transform: uppercase; margin-top: 2px; }
    .fg-mast-right { text-align: right; font-family: 'IBM Plex Mono', monospace; }
    .fg-live { display: inline-flex; align-items: center; gap: 6px; font-size: 0.72rem; letter-spacing: 0.08em; color: var(--c-clear); }
    .fg-live-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--c-clear); animation: fgpulse 2.2s infinite; }
    @keyframes fgpulse { 0% {box-shadow:0 0 0 0 rgba(53,208,127,.5);} 70%{box-shadow:0 0 0 6px rgba(53,208,127,0);} 100%{box-shadow:0 0 0 0 rgba(53,208,127,0);} }
    .fg-mast-time { font-size: 0.95rem; margin-top: 3px; }

    .fg-section-title { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--text-muted);
        margin: 0 0 0.6rem; padding-bottom: 0.4rem; border-bottom: 1px solid var(--line); }

    .fg-kpi-row { display: flex; border: 1px solid var(--line); border-radius: 3px; overflow: hidden; background: var(--bg-panel); margin-bottom: 1.6rem;}
    .fg-kpi-tile { flex: 1; padding: 0.85rem 1.1rem; border-right: 1px solid var(--line); font-family: 'IBM Plex Mono', monospace; }
    .fg-kpi-tile:last-child { border-right: none; }
    .fg-kpi-label { font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-muted); }
    .fg-kpi-value { font-family: 'Rajdhani', sans-serif; font-size: 1.9rem; font-weight: 700; margin-top: 2px; color: var(--text-hi); }
    .fg-kpi-unit { font-size: 0.85rem; color: var(--text-muted); margin-left: 4px; font-family: 'IBM Plex Mono', monospace; }

    .fg-panels { display: flex; gap: 1rem; margin-bottom: 0.6rem; font-family: 'IBM Plex Mono', monospace; }
    .fg-panel { position: relative; flex: 1; background: var(--bg-panel); border: 1px solid var(--line);
        border-radius: 3px; padding: 1rem 1.1rem 1.1rem; }
    .fg-panel::before, .fg-panel::after { content: ""; position: absolute; width: 10px; height: 10px; top: -1px; border-top: 2px solid var(--accent); opacity: .9;}
    .fg-panel::before { left: -1px; border-left: 2px solid var(--accent); }
    .fg-panel::after { right: -1px; border-right: 2px solid var(--accent); }
    .fg-panel-head { display: flex; justify-content: space-between; align-items: baseline; }
    .fg-panel-h { font-family: 'Rajdhani', sans-serif; font-size: 1rem; font-weight: 600; color: var(--text-hi); margin-bottom:0;}
    .fg-badge { font-size: 0.65rem; padding: 2px 7px; border-radius: 20px; border: 1px solid var(--accent); color: var(--accent); }
    .fg-panel-value { font-family: 'Rajdhani', sans-serif; font-size: 2.3rem; font-weight: 700; margin: 0.45rem 0 0; color: var(--text-hi); }
    .fg-panel-range { font-size: 0.8rem; color: var(--text-muted); margin-left: 6px; font-family: 'IBM Plex Mono', monospace; }
    .fg-panel-status { display: flex; align-items: center; gap: 6px; margin-top: 0.6rem; font-size: 0.82rem; color: var(--text-hi);}
    .fg-status-dot { width: 8px; height: 8px; border-radius: 2px; background: var(--accent); }
    .fg-panel-speed { margin-top: 0.35rem; font-size: 0.82rem; color: var(--text-muted); }
    .fg-panel-speed b { color: var(--text-hi); }
    .fg-caption { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: var(--text-muted); margin: 0.4rem 0 1.6rem; line-height: 1.5; max-width: 900px; }

    .fg-box { background: var(--bg-panel); border: 1px solid var(--line); border-radius: 3px; padding: 1rem 1.1rem 0.6rem; margin-bottom: 1.4rem; }
    
    /* Streamlit sidebar styling adjustments */
    [data-testid="stSidebar"] { background-color: var(--bg-panel); border-right: 1px solid var(--line); }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    models = joblib.load("fog_models_v2.joblib")
    artifacts = joblib.load("artifacts_v2.joblib")
    test_df = pd.read_csv("test_features_v2.csv", index_col=0, parse_dates=True)
    return models, artifacts, test_df

models, artifacts, test_df = load_assets()
reg_models, clf_models = models["reg"], models["clf"]
features = artifacts["features"]
CRIT_THRESH, ADV_THRESH = artifacts["crit_thresh"], artifacts["adv_thresh"]
QUANTILES = artifacts["quantiles"]
HORIZONS = artifacts["horizons"]
CLASS_LABELS = artifacts["class_labels"]

# V3 App Styling Definitions
STATUS_STYLE = {
    "CRITICAL": {"label": "Critical risk", "color": "#ff4d4f", "speed": "10 km/h · Convoy Mode"},
    "ADVISORY": {"label": "Moderate advisory", "color": "#f5a623", "speed": "22 km/h"},
    "CLEAR":    {"label": "Clear / normal", "color": "#35d07f", "speed": "40 km/h · Max"},
}
CLASS_TO_STATUS = {0: "CLEAR", 1: "ADVISORY", 2: "CRITICAL"}

# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown("<div style='font-family:Rajdhani; font-size:1.15rem; font-weight:700;'>🛡️ FOGGUARD</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='font-size:0.65rem; letter-spacing:0.1em; color:#838d97; text-transform:uppercase; margin-bottom:1.2rem;'>Central control</div>", unsafe_allow_html=True)

available_dates = sorted(set(test_df.index.date))
default_idx = min(10, len(available_dates) - 1)

if "date_ptr" not in st.session_state:
    st.session_state.date_ptr = default_idx

st.sidebar.markdown("<div style='font-size:0.68rem; letter-spacing:0.12em; text-transform:uppercase; color:#838d97; border-bottom:1px solid #262c33; padding-bottom:0.35rem; margin-bottom:0.6rem;'>Timeline control</div>", unsafe_allow_html=True)

jc1, jc2, jc3 = st.sidebar.columns(3)
if jc1.button("◂ Prev", use_container_width=True):
    st.session_state.date_ptr = max(0, st.session_state.date_ptr - 1)
if jc2.button("Random", use_container_width=True):
    st.session_state.date_ptr = int(np.random.randint(0, len(available_dates)))
if jc3.button("Next ▸", use_container_width=True):
    st.session_state.date_ptr = min(len(available_dates) - 1, st.session_state.date_ptr + 1)

selected_date = st.sidebar.selectbox(
    "Operation date",
    options=available_dates,
    index=st.session_state.date_ptr,
    format_func=lambda d: d.strftime("%d %b %Y"),
)
st.session_state.date_ptr = available_dates.index(selected_date)

day_data = test_df[test_df.index.date == selected_date]
if day_data.empty:
    st.warning("No telemetry available for selected date.")
    st.stop()

valid_times = day_data.index.time
selected_time = st.sidebar.selectbox(
    "Forecast hour", valid_times, index=len(valid_times) // 2,
    format_func=lambda t: t.strftime("%H:%M"),
)
current_dt = pd.to_datetime(f"{selected_date} {selected_time}")
curr_idx = test_df.index.get_loc(current_dt)
current_row = test_df.iloc[[curr_idx]]

st.sidebar.markdown("<div style='font-size:0.68rem; letter-spacing:0.12em; text-transform:uppercase; color:#838d97; border-bottom:1px solid #262c33; padding-bottom:0.35rem; margin-top:1.5rem; margin-bottom:0.6rem;'>Diagnostics</div>", unsafe_allow_html=True)
show_backtest = st.sidebar.checkbox("Show model backtest", value=False)

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
X_curr = current_row[features]
med = np.array([reg_models[h][0.5].predict(X_curr)[0] for h in HORIZONS])
lo = np.array([reg_models[h][QUANTILES[0]].predict(X_curr)[0] for h in HORIZONS])
hi = np.array([reg_models[h][QUANTILES[-1]].predict(X_curr)[0] for h in HORIZONS])
med, lo, hi = (np.clip(a, 0.0, 25.0) for a in (med, lo, hi))

cls_pred = [int(clf_models[h].predict(X_curr)[0]) for h in HORIZONS]
cls_proba = [clf_models[h].predict_proba(X_curr)[0] for h in HORIZONS]

# ---------------------------------------------------------------------------
# Main Dashboard Panel 
# ---------------------------------------------------------------------------
dt_str_date = current_dt.strftime("%d %b %Y")
dt_str_time = current_dt.strftime("%H:%M")

masthead_html = (
    '<div class="fg-mast">'
    '<div>'
    '<p class="fg-mast-title">Dynamic Haul-Road Speed Advisory</p>'
    '<p class="fg-mast-sub">ML fog nowcasting · site operations</p>'
    '</div>'
    '<div class="fg-mast-right">'
    '<div class="fg-live"><span class="fg-live-dot"></span>LIVE FEED · TEST SET REPLAY</div>'
    f'<div class="fg-mast-time">{dt_str_date} &nbsp;&nbsp; {dt_str_time}</div>'
    '</div>'
    '</div>'
)
st.markdown(masthead_html, unsafe_allow_html=True)

st.markdown('<div class="fg-section-title">Current conditions</div>', unsafe_allow_html=True)

kpi_html = (
    '<div class="fg-kpi-row">'
    f'<div class="fg-kpi-tile"><div class="fg-kpi-label">Visibility</div><div class="fg-kpi-value">{current_row["Visibility_km"].values[0]:.1f}<span class="fg-kpi-unit">km</span></div></div>'
    f'<div class="fg-kpi-tile"><div class="fg-kpi-label">Temperature</div><div class="fg-kpi-value">{current_row["Temp_C"].values[0]:.1f}<span class="fg-kpi-unit">°C</span></div></div>'
    f'<div class="fg-kpi-tile"><div class="fg-kpi-label">Rel. humidity</div><div class="fg-kpi-value">{current_row["Rel Hum_%"].values[0]:.0f}<span class="fg-kpi-unit">%</span></div></div>'
    f'<div class="fg-kpi-tile"><div class="fg-kpi-label">Dew deficit</div><div class="fg-kpi-value">{current_row["Dew_Point_Deficit"].values[0]:.1f}<span class="fg-kpi-unit">°C</span></div></div>'
    '</div>'
)
st.markdown(kpi_html, unsafe_allow_html=True)

st.markdown('<div class="fg-section-title">Forecast advisory</div>', unsafe_allow_html=True)

panels_inner = ""
for i, h_label in enumerate(["T+1 hour", "T+2 hours", "T+3 hours"]):
    status = CLASS_TO_STATUS[cls_pred[i]]
    style = STATUS_STYLE[status]
    conf = cls_proba[i][cls_pred[i]] * 100
    
    panels_inner += (
        f'<div class="fg-panel" style="--accent:{style["color"]};">'
        f'<div class="fg-panel-head"><span class="fg-panel-h">{h_label}</span><span class="fg-badge">{conf:.0f}% conf.</span></div>'
        f'<div class="fg-panel-value">{med[i]:.2f} km<span class="fg-panel-range">{lo[i]:.1f}–{hi[i]:.1f}</span></div>'
        f'<div class="fg-panel-status"><span class="fg-status-dot"></span>{style["label"]}</div>'
        f'<div class="fg-panel-speed">Target speed &nbsp;<b>{style["speed"]}</b></div>'
        '</div>'
    )

st.markdown(f'<div class="fg-panels">{panels_inner}</div>', unsafe_allow_html=True)
st.markdown('<div class="fg-caption">Status comes from a dedicated risk classifier (tuned for recall on fog events), not just the raw visibility forecast — it catches hazards the point forecast alone tends to smooth over. Range in parentheses is the p10–p90 forecast band.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Operational Trajectory Chart 
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="fg-box" style="margin-bottom:0; padding-bottom:0; border-bottom-left-radius:0; border-bottom-right-radius:0; border-bottom:0;">'
    '<div class="fg-panel-h" style="margin-bottom:0.4rem;">24-hour operational trajectory</div>'
    '</div>', 
    unsafe_allow_html=True
)

start_idx = max(0, curr_idx - 12)
end_idx = min(len(test_df), curr_idx + 13)
window_data = test_df.iloc[start_idx:end_idx]
future_times = [current_dt + pd.Timedelta(hours=k) for k in range(1, len(HORIZONS) + 1)]

fig = go.Figure()

# Actual Telemetry (V3 Colors)
fig.add_trace(go.Scatter(
    x=window_data.index, y=window_data["Visibility_km"],
    mode="lines+markers", name="Actual ground truth",
    line=dict(color="#5b6672", width=2), marker=dict(size=5, color="#5b6672")
))

# Confidence Band (V3 Colors)
band_x = future_times + future_times[::-1]
band_y = list(hi) + list(lo[::-1])
fig.add_trace(go.Scatter(
    x=band_x, y=band_y, fill="toself",
    fillcolor="rgba(41,211,199,0.13)", line=dict(color="rgba(0,0,0,0)"),
    name="p10–p90 band", hoverinfo="skip"
))

# ML Projection (V3 Colors)
fig.add_trace(go.Scatter(
    x=[current_dt] + future_times, y=[current_row['Visibility_km'].values[0]] + list(med),
    mode="lines+markers", name="ML fog nowcast (median)",
    line=dict(color="#29d3c7", width=3), marker=dict(size=9, symbol="diamond", color="#29d3c7")
))

# Layout update for V3 Theme
fig.update_layout(
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono, monospace", color="#eae7e0", size=12),
    height=400,
    xaxis_title="Hours from present", yaxis_title="Visibility (km)",
    yaxis=dict(range=[0, 26], gridcolor="rgba(255,255,255,0.05)", zeroline=False),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
    hovermode="x unified", margin=dict(l=10, r=10, t=25, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)")
)

# Present Time Marker
fig.add_shape(
    type="line", xref="x", yref="paper",
    x0=current_dt, x1=current_dt, y0=0, y1=1,
    line=dict(color="#ffb020", dash="dash", width=1.5),
)
fig.add_annotation(
    x=current_dt, y=1.05, xref='x', yref='paper',
    text='PRESENT · T+0', showarrow=False,
    font=dict(color='#ffb020', size=11), xanchor='left'
)

st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
st.markdown('<div class="fg-box" style="margin-top:-3rem; padding-top:0; border-top-left-radius:0; border-top-right-radius:0; border-top:0; height:3rem;"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Feature Contribution 
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="fg-box" style="margin-bottom:0; padding-bottom:0; border-bottom-left-radius:0; border-bottom-right-radius:0; border-bottom:0;">'
    '<div class="fg-panel-h">What\'s driving this forecast</div>'
    '<div style="font-size:0.75rem;color:var(--text-muted);margin:0.2rem 0 0.4rem; font-family:\'IBM Plex Mono\', monospace;">Average feature importance across the t+1/t+2/t+3 risk classifiers.</div>'
    '</div>', 
    unsafe_allow_html=True
)

importances = np.mean([clf_models[h].feature_importances_ for h in HORIZONS], axis=0)
imp_df = (
    pd.DataFrame({"feature": features, "importance": importances})
    .sort_values("importance", ascending=False)
    .head(10)
    .iloc[::-1]
)
imp_fig = go.Figure(go.Bar(
    x=imp_df["importance"], y=imp_df["feature"], orientation="h",
    marker_color="#ffb020"
))
imp_fig.update_layout(
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono, monospace", color="#eae7e0", size=12),
    height=330,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(title="Relative importance", gridcolor="rgba(255,255,255,0.05)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.02)")
)
st.plotly_chart(imp_fig, use_container_width=True, config={'displayModeBar': False})
st.markdown('<div class="fg-box" style="margin-top:-3rem; padding-top:0; border-top-left-radius:0; border-top-right-radius:0; border-top:0; height:3rem;"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Backtest Tab 
# ---------------------------------------------------------------------------
if show_backtest:
    st.markdown('<div class="fg-section-title">Model Validation Metrics</div>', unsafe_allow_html=True)

    @st.cache_data
    def compute_backtest():
        rows = []
        X_all = test_df[features]
        for h in HORIZONS:
            y_pred_med = np.clip(reg_models[h][0.5].predict(X_all), 0, 25)
            y_pred_cls = clf_models[h].predict(X_all)
            y_true = test_df["Visibility_km"].shift(-h).values
            mask = ~np.isnan(y_true)
            y_true_m, y_pred_m, y_pred_c = y_true[mask], y_pred_med[mask], y_pred_cls[mask]
            y_true_cls = np.where(y_true_m <= CRIT_THRESH, 2, np.where(y_true_m <= ADV_THRESH, 1, 0))
            rmse = float(np.sqrt(np.mean((y_true_m - y_pred_m) ** 2)))
            mae = float(np.mean(np.abs(y_true_m - y_pred_m)))
            recall_hazard = float(np.mean((y_true_cls[y_true_cls > 0] > 0) &
                                           (y_pred_c[y_true_cls > 0] > 0))) if (y_true_cls > 0).any() else float("nan")
            rows.append({"Prediction Horizon": f"t+{h}", "RMSE (km)": round(rmse, 2), "MAE (km)": round(mae, 2),
                         "Hazard Event Recall": round(recall_hazard, 2)})
        return pd.DataFrame(rows)

    st.dataframe(compute_backtest(), use_container_width=True, hide_index=True)
