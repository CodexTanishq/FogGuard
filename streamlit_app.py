import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

st.set_page_config(page_title="FogGuard | Central Control", page_icon="🛡️", layout="wide")

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

STATUS_STYLE = {
    "CRITICAL": {"label": "CRITICAL RISK", "color": "#ff4b4b", "speed": "10 km/h (Convoy Mode)"},
    "ADVISORY": {"label": "MODERATE ADVISORY", "color": "#ffa500", "speed": "22 km/h"},
    "CLEAR":    {"label": "CLEAR / NORMAL", "color": "#00c04b", "speed": "40 km/h (Max)"},
}
CLASS_TO_STATUS = {0: "CLEAR", 1: "ADVISORY", 2: "CRITICAL"}

# ---------------------------------------------------------------------------
# Sidebar — browse any date/time actually present in the dataset
# ---------------------------------------------------------------------------
st.sidebar.title("🛡️ FogGuard Control")
st.sidebar.markdown("Browse the operational timeline to evaluate ML nowcasting performance.")

available_dates = sorted(set(test_df.index.date))
default_idx = min(10, len(available_dates) - 1)

st.sidebar.markdown("**Jump to**")
jc1, jc2, jc3 = st.sidebar.columns(3)
if "date_ptr" not in st.session_state:
    st.session_state.date_ptr = default_idx
if jc1.button("⏮ Prev day", use_container_width=True):
    st.session_state.date_ptr = max(0, st.session_state.date_ptr - 1)
if jc2.button("Random", use_container_width=True):
    st.session_state.date_ptr = int(np.random.randint(0, len(available_dates)))
if jc3.button("Next day ⏭", use_container_width=True):
    st.session_state.date_ptr = min(len(available_dates) - 1, st.session_state.date_ptr + 1)

selected_date = st.sidebar.selectbox(
    "Select Operation Date",
    options=available_dates,
    index=st.session_state.date_ptr,
    format_func=lambda d: d.strftime("%d %b %Y"),
)
st.session_state.date_ptr = available_dates.index(selected_date)

day_data = test_df[test_df.index.date == selected_date]
if day_data.empty:
    st.warning("No data available for this date. Please select another.")
    st.stop()

valid_times = day_data.index.time
selected_time = st.sidebar.selectbox(
    "Select Forecast Hour", valid_times, index=len(valid_times) // 2,
    format_func=lambda t: t.strftime("%H:%M"),
)
current_dt = pd.to_datetime(f"{selected_date} {selected_time}")
curr_idx = test_df.index.get_loc(current_dt)
current_row = test_df.iloc[[curr_idx]]

st.sidebar.markdown("---")
show_backtest = st.sidebar.checkbox("Show model backtest tab", value=False)

# ---------------------------------------------------------------------------
# Predictions: quantile band (median = point forecast) + classifier risk class
# ---------------------------------------------------------------------------
X_curr = current_row[features]
med = np.array([reg_models[h][0.5].predict(X_curr)[0] for h in HORIZONS])
lo = np.array([reg_models[h][QUANTILES[0]].predict(X_curr)[0] for h in HORIZONS])
hi = np.array([reg_models[h][QUANTILES[-1]].predict(X_curr)[0] for h in HORIZONS])
med, lo, hi = (np.clip(a, 0.0, 25.0) for a in (med, lo, hi))

cls_pred = [int(clf_models[h].predict(X_curr)[0]) for h in HORIZONS]
cls_proba = [clf_models[h].predict_proba(X_curr)[0] for h in HORIZONS]

# ---------------------------------------------------------------------------
# Header + current conditions
# ---------------------------------------------------------------------------
st.title("Dynamic Haul-Road Speed Advisory")
curr_vis = current_row["Visibility_km"].values[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Visibility", f"{curr_vis:.1f} km")
c2.metric("Temperature", f"{current_row['Temp_C'].values[0]:.1f} °C")
c3.metric("Rel. Humidity", f"{current_row['Rel Hum_%'].values[0]:.0f} %")
c4.metric("Dew Deficit", f"{current_row['Dew_Point_Deficit'].values[0]:.1f} °C")

st.markdown("---")
h_cols = st.columns(3)
for i, (col, h_label) in enumerate(zip(h_cols, ["T+1 Hour Ahead", "T+2 Hours Ahead", "T+3 Hours Ahead"])):
    status = CLASS_TO_STATUS[cls_pred[i]]
    style = STATUS_STYLE[status]
    conf = cls_proba[i][cls_pred[i]] * 100
    with col:
        st.markdown(f"""
        <div style="background-color:#1e2530; padding:15px; border-radius:8px; border-left:5px solid {style['color']};">
            <h4 style="margin:0; color:{style['color']};">{h_label}</h4>
            <h2 style="margin:5px 0;">{med[i]:.2f} km <span style="font-size:14px; color:#9aa4b2;">({lo[i]:.1f}–{hi[i]:.1f})</span></h2>
            <p style="margin:2px 0;"><b>Status:</b> {style['label']} <span style="color:#9aa4b2;">({conf:.0f}% conf.)</span></p>
            <p style="margin:2px 0;"><b>Target Speed:</b> {style['speed']}</p>
        </div>
        """, unsafe_allow_html=True)
st.caption("Status comes from a dedicated risk classifier (tuned for recall on fog events), not just the raw visibility forecast — it catches hazards the point forecast alone tends to smooth over. Range in parentheses is the p10–p90 forecast band.")

# ---------------------------------------------------------------------------
# 24-hour trajectory chart with confidence band
# ---------------------------------------------------------------------------
st.markdown("### 24-Hour Operational Trajectory")
start_idx = max(0, curr_idx - 12)
end_idx = min(len(test_df), curr_idx + 13)
window_data = test_df.iloc[start_idx:end_idx]
future_times = [current_dt + pd.Timedelta(hours=k) for k in range(1, len(HORIZONS) + 1)]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=window_data.index, y=window_data["Visibility_km"],
    mode="lines+markers", name="Actual Ground Truth",
    line=dict(color="#4b5563", width=2), marker=dict(size=6, color="#4b5563"),
))

# p10-p90 confidence band
band_x = future_times + future_times[::-1]
band_y = list(hi) + list(lo[::-1])
fig.add_trace(go.Scatter(
    x=band_x, y=band_y, fill="toself",
    fillcolor="rgba(0,229,255,0.15)", line=dict(color="rgba(0,0,0,0)"),
    name="p10–p90 band", hoverinfo="skip",
))

fig.add_trace(go.Scatter(
    x=[current_dt] + future_times, y=[curr_vis] + list(med),
    mode="lines+markers", name="ML Fog Nowcast (median)",
    line=dict(color="#00e5ff", width=3), marker=dict(size=10, symbol="diamond", color="#00e5ff"),
))

fig.add_hrect(y0=0, y1=CRIT_THRESH, fillcolor="red", opacity=0.15, line_width=0, annotation_text="Critical Brake Zone")
fig.add_hrect(y0=CRIT_THRESH, y1=ADV_THRESH, fillcolor="orange", opacity=0.1, line_width=0, annotation_text="Advisory Speed Zone")

fig.update_layout(
    template="plotly_dark", height=430,
    xaxis_title="Timeline", yaxis_title="Visibility (km)",
    yaxis=dict(range=[0, 26]),
    hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10),
)

# "Present" marker — add_shape/add_annotation instead of add_vline, which
# avoids a Plotly bug where add_vline's auto label-centering math breaks on
# some x-axis/value combinations.
fig.add_shape(
    type="line", xref="x", yref="paper",
    x0=current_dt, x1=current_dt, y0=0, y1=1,
    line=dict(color="white", dash="dash", width=1.5),
)
fig.add_annotation(
    x=current_dt, y=1.04, xref="x", yref="paper",
    text="Present (T=0)", showarrow=False,
    font=dict(color="white", size=11), xanchor="left",
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Feature importance — what's driving the risk classification right now
# ---------------------------------------------------------------------------
st.markdown("### What's Driving This Forecast")
st.caption("Average feature importance across the t+1/t+2/t+3 risk classifiers — a transparent 'why' behind the advisory, not a black box.")

importances = np.mean([clf_models[h].feature_importances_ for h in HORIZONS], axis=0)
imp_df = (
    pd.DataFrame({"feature": features, "importance": importances})
    .sort_values("importance", ascending=False)
    .head(10)
    .iloc[::-1]
)
imp_fig = go.Figure(go.Bar(
    x=imp_df["importance"], y=imp_df["feature"], orientation="h",
    marker_color="#00e5ff",
))
imp_fig.update_layout(
    template="plotly_dark", height=350,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="Relative importance",
)
st.plotly_chart(imp_fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Optional backtest tab — model vs persistence over the WHOLE held-out set
# ---------------------------------------------------------------------------
if show_backtest:
    st.markdown("---")
    st.markdown("### Model Backtest (full held-out test set)")

    @st.cache_data
    def compute_backtest():
        rows = []
        X_all = test_df[features]
        for h in HORIZONS:
            y_pred_med = np.clip(reg_models[h][0.5].predict(X_all), 0, 25)
            y_pred_cls = clf_models[h].predict(X_all)
            # true future visibility for this horizon (shift within this df)
            y_true = test_df["Visibility_km"].shift(-h).values
            mask = ~np.isnan(y_true)
            y_true_m, y_pred_m, y_pred_c = y_true[mask], y_pred_med[mask], y_pred_cls[mask]
            y_true_cls = np.where(y_true_m <= CRIT_THRESH, 2, np.where(y_true_m <= ADV_THRESH, 1, 0))
            rmse = float(np.sqrt(np.mean((y_true_m - y_pred_m) ** 2)))
            mae = float(np.mean(np.abs(y_true_m - y_pred_m)))
            recall_hazard = float(np.mean((y_true_cls[y_true_cls > 0] > 0) &
                                           (y_pred_c[y_true_cls > 0] > 0))) if (y_true_cls > 0).any() else float("nan")
            rows.append({"Horizon": f"t+{h}", "RMSE": round(rmse, 2), "MAE": round(mae, 2),
                         "Fog Event Recall": round(recall_hazard, 2)})
        return pd.DataFrame(rows)

    st.dataframe(compute_backtest(), use_container_width=True, hide_index=True)
    st.caption("Fog Event Recall = share of true ADVISORY/CRITICAL hours the classifier correctly flagged as at-risk (any non-CLEAR class).")