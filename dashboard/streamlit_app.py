"""
Streamlit Anomaly Detection Dashboard
=======================================
Visual monitoring dashboard that:
  - Calls your FastAPI /predict/batch endpoint in real time
  - Shows live anomaly metrics and trends
  - Displays per-endpoint health status
  - Auto-refreshes every 10 seconds

Run: streamlit run dashboard/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import random
import time
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# Switch API_URL to your Render URL once deployed
# ─────────────────────────────────────────────────────────────────────────────
#API_URL = "http://localhost:8000"   # local
API_URL = "https://api-anomaly-detector.onrender.com"  # production

ENDPOINTS = [
    "/api/v1/customers",
    "/api/v1/orders",
    "/api/v1/payments",
    "/api/v1/inventory",
    "/api/v2/products",
    "/api/v1/shipments",
    "/api/v1/auth/token",
    "/api/v2/invoices",
]

REFRESH_INTERVAL = 10   # seconds between auto-refresh

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "API Anomaly Detector",
    page_icon  = "radar",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — makes it look professional
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .critical { color: #ff4b4b; font-weight: bold; }
    .high     { color: #ffa500; font-weight: bold; }
    .medium   { color: #ffd700; font-weight: bold; }
    .low      { color: #00cc44; font-weight: bold; }
    .header-title {
        font-size: 2rem;
        font-weight: bold;
        color: #7c3aed;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA GENERATORS
# Simulates realistic API metrics to send to the model
# In production, this is replaced by real Anypoint telemetry
# ─────────────────────────────────────────────────────────────────────────────
def generate_normal_metrics(endpoint_id):
    """Generates a healthy API call record."""
    return {
        "endpoint_id"             : endpoint_id,
        "response_time_ms"        : round(random.gauss(180, 40), 2),
        "payload_size_kb"         : round(random.uniform(1, 50), 2),
        "is_error"                : 0,
        "is_4xx"                  : 0,
        "rolling_avg_response_ms" : round(random.gauss(178, 20), 2),
        "rolling_error_rate"      : round(random.uniform(0, 0.05), 3),
        "rolling_avg_payload_kb"  : round(random.uniform(10, 40), 2),
        "response_time_deviation" : round(random.uniform(0, 30), 2),
        "hour_of_day"             : datetime.now().hour,
        "day_of_week"             : datetime.now().weekday()
    }

def generate_anomaly_metrics(endpoint_id):
    """Generates an anomalous API call record."""
    anomaly_type = random.choice(["latency", "error", "flood"])

    base = {
        "endpoint_id" : endpoint_id,
        "is_4xx"      : 0,
        "hour_of_day" : datetime.now().hour,
        "day_of_week" : datetime.now().weekday()
    }

    if anomaly_type == "latency":
        base.update({
            "response_time_ms"        : round(random.uniform(3000, 8000), 2),
            "payload_size_kb"         : round(random.uniform(1, 50), 2),
            "is_error"                : 0,
            "rolling_avg_response_ms" : round(random.gauss(178, 20), 2),
            "rolling_error_rate"      : 0.0,
            "rolling_avg_payload_kb"  : round(random.uniform(10, 40), 2),
            "response_time_deviation" : round(random.uniform(2000, 7000), 2),
        })
    elif anomaly_type == "error":
        base.update({
            "response_time_ms"        : round(random.gauss(300, 60), 2),
            "payload_size_kb"         : round(random.uniform(0.1, 5), 2),
            "is_error"                : 1,
            "rolling_avg_response_ms" : round(random.gauss(178, 20), 2),
            "rolling_error_rate"      : round(random.uniform(0.5, 1.0), 3),
            "rolling_avg_payload_kb"  : round(random.uniform(1, 10), 2),
            "response_time_deviation" : round(random.uniform(50, 200), 2),
        })
    elif anomaly_type == "flood":
        base.update({
            "response_time_ms"        : round(random.gauss(400, 100), 2),
            "payload_size_kb"         : round(random.uniform(200, 500), 2),
            "is_error"                : 0,
            "rolling_avg_response_ms" : round(random.gauss(178, 20), 2),
            "rolling_error_rate"      : round(random.uniform(0, 0.1), 3),
            "rolling_avg_payload_kb"  : round(random.uniform(150, 400), 2),
            "response_time_deviation" : round(random.uniform(100, 500), 2),
        })

    return base

def generate_batch(size=20, anomaly_rate=0.15):
    """Generates a mixed batch of normal + anomalous records."""
    records = []
    for _ in range(size):
        endpoint_id = random.randint(0, 7)
        if random.random() < anomaly_rate:
            records.append(generate_anomaly_metrics(endpoint_id))
        else:
            records.append(generate_normal_metrics(endpoint_id))
    return records


# ─────────────────────────────────────────────────────────────────────────────
# API CALLER
# ─────────────────────────────────────────────────────────────────────────────
def call_predict_batch(records):
    """Sends batch to FastAPI and returns predictions."""
    try:
        response = requests.post(
            f"{API_URL}/predict/batch",
            json    = records,
            timeout = 10
        )
        if response.status_code == 200:
            return response.json(), None
        return None, f"API Error: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Is the server running?"
    except Exception as e:
        return None, str(e)

def check_health():
    """Checks if the FastAPI server is up."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200
    except:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# Streamlit re-runs the entire script on each interaction
# Session state persists data across re-runs — like a global variable
# ─────────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history        = []   # list of all predictions
if "total_calls" not in st.session_state:
    st.session_state.total_calls    = 0
if "total_anomalies" not in st.session_state:
    st.session_state.total_anomalies = 0
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh   = datetime.now()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")

    api_url_input = st.text_input(
        "API URL",
        value = API_URL,
        help  = "Switch to your Render URL for production"
    )
    if api_url_input != API_URL:
        API_URL = api_url_input

    batch_size   = st.slider("Batch size per refresh", 10, 50, 20)
    anomaly_rate = st.slider("Simulated anomaly rate", 0.05, 0.40, 0.15)
    auto_refresh = st.toggle("Auto Refresh (10s)", value=False)

    st.divider()

    # API Health indicator
    st.markdown("### API Health")
    is_healthy = check_health()
    if is_healthy:
        st.success("API is Online")
    else:
        st.error("API is Offline")
        st.info("Start with: uvicorn serving.app:app --reload --port 8000")

    st.divider()

    if st.button("Reset Dashboard", type="secondary"):
        st.session_state.history         = []
        st.session_state.total_calls     = 0
        st.session_state.total_anomalies = 0
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="header-title">API Anomaly Detector — Live Dashboard</div>',
    unsafe_allow_html=True
)
st.markdown(f"MuleSoft API Intelligence Platform | Last updated: {datetime.now().strftime('%H:%M:%S')}")
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# FETCH NEW DATA
# ─────────────────────────────────────────────────────────────────────────────
col_btn1, col_btn2, col_spacer = st.columns([1, 1, 6])

with col_btn1:
    run_now = st.button("Run Detection", type="primary", use_container_width=True)
with col_btn2:
    st.button("Refresh", use_container_width=True)

if run_now or auto_refresh:
    with st.spinner("Calling ML model..."):
        records = generate_batch(size=batch_size, anomaly_rate=anomaly_rate)
        result, error = call_predict_batch(records)

        if error:
            st.error(f"Error: {error}")
        elif result:
            # Store results in session history
            now = datetime.now()
            for i, (record, prediction) in enumerate(zip(records, result["results"])):
                st.session_state.history.append({
                    "timestamp"     : now.strftime("%H:%M:%S"),
                    "endpoint"      : ENDPOINTS[record["endpoint_id"]],
                    "response_ms"   : record["response_time_ms"],
                    "is_anomaly"    : prediction["is_anomaly"],
                    "anomaly_score" : prediction["anomaly_score"],
                    "risk_level"    : prediction["risk_level"],
                    "message"       : prediction["message"]
                })

            st.session_state.total_calls     += result["total_records"]
            st.session_state.total_anomalies += result["anomaly_count"]
            st.session_state.last_refresh     = now

            # Keep only last 200 records to avoid memory bloat
            if len(st.session_state.history) > 200:
                st.session_state.history = st.session_state.history[-200:]

            st.success(f"Scored {result['total_records']} API calls — {result['anomaly_count']} anomalies detected")


# ─────────────────────────────────────────────────────────────────────────────
# LIVE METRICS ROW
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Live Metrics")
m1, m2, m3, m4 = st.columns(4)

anomaly_rate_pct = (
    (st.session_state.total_anomalies / st.session_state.total_calls * 100)
    if st.session_state.total_calls > 0 else 0
)

m1.metric("Total API Calls Scored",  st.session_state.total_calls)
m2.metric("Total Anomalies Detected", st.session_state.total_anomalies)
m3.metric("Anomaly Rate",            f"{anomaly_rate_pct:.1f}%")
m4.metric(
    "Status",
    "ALERT" if anomaly_rate_pct > 10 else "HEALTHY",
    delta = "Above threshold!" if anomaly_rate_pct > 10 else "Within normal range"
)


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS + ENDPOINT HEALTH
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)

    st.divider()
    chart_col, health_col = st.columns([2, 1])

    # ── Anomaly Score Trend Chart ──────────────────────────────────────────
    with chart_col:
        st.markdown("### Anomaly Score Trend")
        chart_df = df[["timestamp", "anomaly_score", "is_anomaly"]].copy()
        chart_df["color"] = chart_df["is_anomaly"].apply(
            lambda x: "Anomaly" if x == 1 else "Normal"
        )
        st.line_chart(
            chart_df.set_index("timestamp")["anomaly_score"],
            use_container_width=True,
            height=300
        )

    # ── Endpoint Health Table ──────────────────────────────────────────────
    with health_col:
        st.markdown("### Endpoint Health")
        endpoint_stats = (
            df.groupby("endpoint")
            .agg(
                total    = ("is_anomaly", "count"),
                anomalies= ("is_anomaly", "sum")
            )
            .reset_index()
        )
        endpoint_stats["rate"] = (
            endpoint_stats["anomalies"] / endpoint_stats["total"] * 100
        ).round(1)
        endpoint_stats["health"] = endpoint_stats["rate"].apply(
            lambda x: "CRITICAL" if x > 20 else "HIGH" if x > 10 else "LOW"
        )
        endpoint_stats["endpoint"] = endpoint_stats["endpoint"].str.replace("/api/", "")

        for _, row in endpoint_stats.iterrows():
            color = "critical" if row["health"] == "CRITICAL" else \
                    "high"     if row["health"] == "HIGH"     else "low"
            icon  = "" if row["health"] == "CRITICAL" else \
                    "" if row["health"] == "HIGH"      else " "
            st.markdown(
                f"{icon} `{row['endpoint']}` — "
                f"<span class='{color}'>{row['health']} ({row['rate']}%)</span>",
                unsafe_allow_html=True
            )

    # ── Risk Level Breakdown ────────────────────────────────────────────────
    st.divider()
    risk_col, feed_col = st.columns([1, 2])

    with risk_col:
        st.markdown("### Risk Breakdown")
        risk_counts = df["risk_level"].value_counts()
        st.bar_chart(risk_counts, use_container_width=True, height=250)

    # ── Recent Anomalies Feed ──────────────────────────────────────────────
    with feed_col:
        st.markdown("### Recent Anomalies Feed")
        anomalies_df = df[df["is_anomaly"] == 1].tail(8)

        if len(anomalies_df) == 0:
            st.info("No anomalies detected yet. Click Run Detection.")
        else:
            for _, row in anomalies_df.iloc[::-1].iterrows():
                color = "critical" if row["risk_level"] == "CRITICAL" else \
                        "high"     if row["risk_level"] == "HIGH"      else "medium"
                icon  = "" if row["risk_level"] == "CRITICAL" else \
                        "" if row["risk_level"] == "HIGH"      else ""
                st.markdown(
                    f"{icon} **{row['timestamp']}** | `{row['endpoint']}` | "
                    f"<span class='{color}'>{row['risk_level']}</span> | "
                    f"Score: {row['anomaly_score']:.2f} | "
                    f"{row['response_ms']:.0f}ms",
                    unsafe_allow_html=True
                )

else:
    st.info("Click 'Run Detection' to start scoring API calls and see live results here.")


# ─────────────────────────────────────────────────────────────────────────────
# AUTO REFRESH
# ─────────────────────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(REFRESH_INTERVAL)
    st.rerun()