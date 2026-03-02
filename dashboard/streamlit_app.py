"""
Fraud Detection - Live Monitoring Dashboard
============================================
Real-time fraud monitoring dashboard that:
  - Scores live transactions against the fraud API
  - Tracks financial impact ($ saved vs $ at risk)
  - Shows per-user fraud profiles
  - Allows live threshold adjustment
  - Exports fraud reports as CSV

Run: streamlit run dashboard/streamlit_app.py
"""

import streamlit as st
import pandas    as pd
import numpy     as np
import requests
import random
import time
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
API_URL = "https://fraud-detector-1pje.onrender.com"
# API_URL = "https://fraud-detector.onrender.com"  # switch for production

MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "gas_station", "pharmacy",
    "clothing", "utilities", "healthcare", "entertainment",
    "electronics", "jewelry", "gift_cards", "luxury_goods",
    "crypto_exchange", "wire_transfer"
]

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Fraud Detection Dashboard",
    page_icon  = "shield",
    layout     = "wide"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #dc2626;
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #1f2937;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border-left: 4px solid #dc2626;
    }
    .safe-card {
        border-left: 4px solid #16a34a !important;
    }
    .warn-card {
        border-left: 4px solid #f59e0b !important;
    }
    .critical { color: #dc2626; font-weight: bold; }
    .high     { color: #f97316; font-weight: bold; }
    .medium   { color: #f59e0b; font-weight: bold; }
    .low      { color: #16a34a; font-weight: bold; }
    .minimal  { color: #6b7280; font-weight: bold; }
    .block-badge  { background:#dc2626; color:white; padding:2px 8px; border-radius:4px; font-size:0.8rem; }
    .review-badge { background:#f59e0b; color:white; padding:2px 8px; border-radius:4px; font-size:0.8rem; }
    .approve-badge{ background:#16a34a; color:white; padding:2px 8px; border-radius:4px; font-size:0.8rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TRANSACTION GENERATORS
# Simulates realistic bank transactions for the demo
# ─────────────────────────────────────────────────────────────────────────────
def gen_legit(user_id, threshold):
    avg = random.uniform(30, 300)
    amt = round(abs(random.gauss(avg, avg * 0.3)), 2)
    return {
        "amount"                : amt,
        "merchant_category"     : random.choice([
            "grocery", "restaurant", "gas_station",
            "pharmacy", "clothing", "entertainment"
        ]),
        "is_international"      : 0,
        "card_present"          : random.choices([1, 0], weights=[80, 20])[0],
        "user_id"               : user_id,
        "amount_vs_user_avg"    : round(amt / avg, 3),
        "transactions_last_1h"  : random.randint(0, 3),
        "transactions_last_24h" : random.randint(1, 8),
        "distance_from_home_km" : round(random.uniform(0, 30), 2),
        "failed_attempts"       : random.choices([0, 1], weights=[92, 8])[0],
        "device_match"          : random.choices([1, 0], weights=[90, 10])[0],
        "hour_of_day"           : random.randint(8, 21),
        "threshold"             : threshold
    }

def gen_fraud(user_id, threshold):
    fraud_type = random.choice([
        "velocity", "geo", "amount", "merchant", "night", "online"
    ])
    avg = random.uniform(30, 200)

    base = {
        "user_id"  : user_id,
        "threshold": threshold
    }

    if fraud_type == "velocity":
        amt = round(random.uniform(10, 100), 2)
        base.update({
            "amount"                : amt,
            "merchant_category"     : "gift_cards",
            "is_international"      : 0,
            "card_present"          : 0,
            "amount_vs_user_avg"    : round(amt / avg, 3),
            "transactions_last_1h"  : random.randint(8, 20),
            "transactions_last_24h" : random.randint(20, 50),
            "distance_from_home_km" : round(random.uniform(0, 15), 2),
            "failed_attempts"       : random.randint(3, 8),
            "device_match"          : 0,
            "hour_of_day"           : random.randint(0, 6),
        })
    elif fraud_type == "geo":
        amt = round(random.uniform(avg * 2, avg * 8), 2)
        base.update({
            "amount"                : amt,
            "merchant_category"     : "electronics",
            "is_international"      : 1,
            "card_present"          : 0,
            "amount_vs_user_avg"    : round(amt / avg, 3),
            "transactions_last_1h"  : random.randint(1, 3),
            "transactions_last_24h" : random.randint(2, 8),
            "distance_from_home_km" : round(random.uniform(3000, 12000), 2),
            "failed_attempts"       : random.randint(1, 3),
            "device_match"          : 0,
            "hour_of_day"           : random.randint(1, 6),
        })
    elif fraud_type == "amount":
        amt = round(random.uniform(avg * 10, avg * 50), 2)
        base.update({
            "amount"                : amt,
            "merchant_category"     : "jewelry",
            "is_international"      : random.choice([0, 1]),
            "card_present"          : 0,
            "amount_vs_user_avg"    : round(amt / avg, 3),
            "transactions_last_1h"  : random.randint(1, 4),
            "transactions_last_24h" : random.randint(1, 6),
            "distance_from_home_km" : round(random.uniform(10, 500), 2),
            "failed_attempts"       : random.randint(1, 5),
            "device_match"          : random.choice([0, 1]),
            "hour_of_day"           : random.randint(0, 23),
        })
    elif fraud_type == "merchant":
        amt = round(random.uniform(avg * 3, avg * 15), 2)
        base.update({
            "amount"                : amt,
            "merchant_category"     : random.choice(["crypto_exchange", "wire_transfer", "luxury_goods"]),
            "is_international"      : 1,
            "card_present"          : 0,
            "amount_vs_user_avg"    : round(amt / avg, 3),
            "transactions_last_1h"  : random.randint(1, 5),
            "transactions_last_24h" : random.randint(2, 10),
            "distance_from_home_km" : round(random.uniform(100, 8000), 2),
            "failed_attempts"       : random.randint(2, 6),
            "device_match"          : 0,
            "hour_of_day"           : random.randint(0, 23),
        })
    elif fraud_type == "night":
        amt = round(random.uniform(avg * 2, avg * 10), 2)
        base.update({
            "amount"                : amt,
            "merchant_category"     : "electronics",
            "is_international"      : 0,
            "card_present"          : 0,
            "amount_vs_user_avg"    : round(amt / avg, 3),
            "transactions_last_1h"  : random.randint(2, 6),
            "transactions_last_24h" : random.randint(3, 12),
            "distance_from_home_km" : round(random.uniform(5, 100), 2),
            "failed_attempts"       : random.randint(2, 5),
            "device_match"          : 0,
            "hour_of_day"           : random.randint(2, 5),
        })
    else:  # online
        amt = round(random.uniform(avg * 2, avg * 12), 2)
        base.update({
            "amount"                : amt,
            "merchant_category"     : "gift_cards",
            "is_international"      : 1,
            "card_present"          : 0,
            "amount_vs_user_avg"    : round(amt / avg, 3),
            "transactions_last_1h"  : random.randint(3, 8),
            "transactions_last_24h" : random.randint(5, 15),
            "distance_from_home_km" : round(random.uniform(500, 10000), 2),
            "failed_attempts"       : random.randint(2, 6),
            "device_match"          : 0,
            "hour_of_day"           : random.randint(0, 23),
        })

    return base

def generate_batch(size, fraud_rate, threshold):
    records = []
    for _ in range(size):
        user_id = random.randint(1, 1000)
        if random.random() < fraud_rate:
            records.append(gen_fraud(user_id, threshold))
        else:
            records.append(gen_legit(user_id, threshold))
    return records


# ─────────────────────────────────────────────────────────────────────────────
# API HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def call_api(endpoint, payload):
    try:
        r = requests.post(
            f"{API_URL}/{endpoint}",
            json    = payload,
            timeout = 15
        )
        if r.status_code == 200:
            return r.json(), None
        return None, f"API Error {r.status_code}: {r.text}"
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Is the server running?"
    except Exception as e:
        return None, str(e)

def check_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200
    except:
        return False

def get_model_info():
    try:
        r = requests.get(f"{API_URL}/model/info", timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
defaults = {
    "history"           : [],
    "total_txns"        : 0,
    "total_fraud"       : 0,
    "total_amount"      : 0.0,
    "total_fraud_amount": 0.0,
    "amount_saved"      : 0.0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Fraud Detection Settings")

    threshold    = st.slider(
        "Fraud Threshold",
        min_value = 0.1,
        max_value = 0.9,
        value     = 0.5,
        step      = 0.05,
        help      = "Lower = catch more fraud but more false alarms"
    )
    batch_size   = st.slider("Transactions per scan", 5, 50, 15)
    fraud_rate   = st.slider("Simulated fraud rate", 0.05, 0.50, 0.20)
    auto_refresh = st.toggle("Auto Scan (10s)", value=False)

    st.divider()

    # API Status
    st.markdown("### API Status")
    is_healthy  = check_health()
    model_info  = get_model_info()

    if is_healthy:
        st.success("Fraud API Online")
    else:
        st.error("Fraud API Offline")
        st.code("uvicorn serving.app:app --reload --port 8000")

    if model_info:
        st.markdown(f"**Model:** {model_info['model_name']}")
        st.markdown(f"**F1 Score:** {model_info['f1_score']}")
        st.markdown(f"**ROC-AUC:** {model_info['roc_auc']}")
        st.markdown(f"**Inference:** {model_info['inference_ms']}ms")

    st.divider()

    if st.button("Reset Dashboard", type="secondary", use_container_width=True):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()

    # CSV Export
    if st.session_state.history:
        df_export = pd.DataFrame(st.session_state.history)
        fraud_only = df_export[df_export["is_fraud"] == 1]
        if len(fraud_only) > 0:
            st.download_button(
                label     = "Download Fraud Report (CSV)",
                data      = fraud_only.to_csv(index=False),
                file_name = f"fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime      = "text/csv",
                use_container_width=True
            )


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Fraud Detection — Live Monitor</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="subtitle">Real-time ML scoring | Threshold: {threshold} | '
    f'Updated: {datetime.now().strftime("%H:%M:%S")}</div>',
    unsafe_allow_html=True
)
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# ACTION BUTTONS
# ─────────────────────────────────────────────────────────────────────────────
btn1, btn2, btn3, spacer = st.columns([1, 1, 1, 5])
with btn1:
    scan_now = st.button("Scan Transactions", type="primary", use_container_width=True)
with btn2:
    st.button("Refresh", use_container_width=True)
with btn3:
    # Single explain test
    explain_test = st.button("Test Explain", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# EXPLAIN TEST — show /predict/explain output
# ─────────────────────────────────────────────────────────────────────────────
if explain_test:
    fraud_payload = {
        "amount"                : 4850.00,
        "merchant_category"     : "electronics",
        "is_international"      : 1,
        "card_present"          : 0,
        "user_id"               : 347,
        "amount_vs_user_avg"    : 18.4,
        "transactions_last_1h"  : 12,
        "transactions_last_24h" : 25,
        "distance_from_home_km" : 8472.3,
        "failed_attempts"       : 5,
        "device_match"          : 0,
        "hour_of_day"           : 3,
        "threshold"             : threshold
    }
    result, error = call_api("predict/explain", fraud_payload)
    if result:
        with st.expander("Explainable AI Result — Why This Transaction Was Flagged", expanded=True):
            col_pred, col_signals = st.columns([1, 2])
            with col_pred:
                st.markdown(f"**Decision:** `{result['prediction']['decision']}`")
                st.markdown(f"**Fraud Probability:** `{result['prediction']['fraud_probability']*100:.1f}%`")
                st.markdown(f"**Risk Level:** `{result['prediction']['risk_level']}`")
                st.markdown(f"**Inference:** `{result['prediction']['inference_ms']}ms`")
            with col_signals:
                st.markdown("**Fraud Signals Detected:**")
                for sig in result["top_fraud_signals"]:
                    severity_color = "critical" if sig["severity"] == "CRITICAL" else \
                                     "high"     if sig["severity"] == "HIGH"      else "medium"
                    st.markdown(
                        f"- <span class='{severity_color}'>{sig['signal']}</span>: {sig['detail']}",
                        unsafe_allow_html=True
                    )
            st.info(result["recommendation"])
    else:
        st.error(f"Error: {error}")


# ─────────────────────────────────────────────────────────────────────────────
# SCAN TRANSACTIONS
# ─────────────────────────────────────────────────────────────────────────────
if scan_now or auto_refresh:
    with st.spinner(f"Scanning {batch_size} transactions..."):
        transactions = generate_batch(batch_size, fraud_rate, threshold)
        result, error = call_api("predict/batch", transactions)

        if error:
            st.error(f"Error: {error}")
        elif result:
            now = datetime.now()

            for txn, pred in zip(transactions, result["results"]):
                st.session_state.history.append({
                    "timestamp"         : now.strftime("%H:%M:%S"),
                    "user_id"           : txn["user_id"],
                    "amount"            : txn["amount"],
                    "merchant_category" : txn["merchant_category"],
                    "is_international"  : txn["is_international"],
                    "hour_of_day"       : txn["hour_of_day"],
                    "failed_attempts"   : txn["failed_attempts"],
                    "device_match"      : txn["device_match"],
                    "is_fraud"          : pred["is_fraud"],
                    "fraud_probability" : pred["fraud_probability"],
                    "risk_level"        : pred["risk_level"],
                    "decision"          : pred["decision"],
                    "inference_ms"      : pred["inference_ms"],
                })

            # Update session counters
            st.session_state.total_txns         += result["total_transactions"]
            st.session_state.total_fraud        += result["fraud_count"]
            st.session_state.total_amount       += sum(t["amount"] for t in transactions)
            st.session_state.total_fraud_amount += result["total_amount_at_risk"]
            st.session_state.amount_saved       += result["total_amount_at_risk"]

            # Keep last 300 records
            if len(st.session_state.history) > 300:
                st.session_state.history = st.session_state.history[-300:]

            if result["fraud_count"] > 0:
                st.error(
                    f"ALERT: {result['fraud_count']} fraud transactions detected! "
                    f"${result['total_amount_at_risk']:,.2f} at risk — BLOCKED"
                )
            else:
                st.success(f"All {result['total_transactions']} transactions appear legitimate")


# ─────────────────────────────────────────────────────────────────────────────
# LIVE METRICS ROW
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Live Financial Metrics")
m1, m2, m3, m4, m5 = st.columns(5)

fraud_rate_pct = (
    st.session_state.total_fraud / st.session_state.total_txns * 100
    if st.session_state.total_txns > 0 else 0
)

m1.metric(
    "Total Scanned",
    f"{st.session_state.total_txns:,}",
)
m2.metric(
    "Fraud Blocked",
    f"{st.session_state.total_fraud:,}",
    delta=f"{fraud_rate_pct:.1f}% rate"
)
m3.metric(
    "Amount Processed",
    f"${st.session_state.total_amount:,.0f}",
)
m4.metric(
    "Amount at Risk (Blocked)",
    f"${st.session_state.total_fraud_amount:,.0f}",
    delta="Saved from fraud" if st.session_state.total_fraud_amount > 0 else None
)
m5.metric(
    "System Status",
    "ALERT" if fraud_rate_pct > 15 else "NORMAL",
    delta="High fraud rate!" if fraud_rate_pct > 15 else "Within threshold"
)


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS + ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.divider()

    # ── ROW 1: Probability Trend + Decision Breakdown ──────────────────────
    chart_col, decision_col = st.columns([2, 1])

    with chart_col:
        st.markdown("### Fraud Probability Trend")
        chart_data = df[["timestamp", "fraud_probability"]].copy()
        chart_data["fraud_threshold"] = threshold
        st.line_chart(
            chart_data.set_index("timestamp")[["fraud_probability", "fraud_threshold"]],
            use_container_width=True,
            height=280
        )
        st.caption("Red line = fraud probability per transaction | Blue line = threshold")

    with decision_col:
        st.markdown("### Decision Breakdown")
        decision_counts = df["decision"].value_counts()
        st.bar_chart(decision_counts, use_container_width=True, height=280)

        total = len(df)
        for decision, count in decision_counts.items():
            badge_class = (
                "block-badge"  if decision == "BLOCK"  else
                "review-badge" if decision == "REVIEW" else
                "approve-badge"
            )
            pct = count / total * 100
            st.markdown(
                f'<span class="{badge_class}">{decision}</span> '
                f'{count} transactions ({pct:.1f}%)',
                unsafe_allow_html=True
            )

    st.divider()

    # ── ROW 2: Merchant Analysis + Risk Distribution ───────────────────────
    merch_col, risk_col = st.columns([1, 1])

    with merch_col:
        st.markdown("### Fraud by Merchant Category")
        merchant_stats = (
            df.groupby("merchant_category")
            .agg(
                total = ("is_fraud", "count"),
                fraud = ("is_fraud", "sum"),
                amount= ("amount",   "sum")
            )
            .reset_index()
        )
        merchant_stats["fraud_rate"] = (
            merchant_stats["fraud"] / merchant_stats["total"] * 100
        ).round(1)
        merchant_stats = merchant_stats.sort_values("fraud_rate", ascending=False)

        for _, row in merchant_stats.head(8).iterrows():
            rate  = row["fraud_rate"]
            color = "critical" if rate > 40 else \
                    "high"     if rate > 20 else \
                    "medium"   if rate > 10 else "low"
            bar   = "█" * int(rate / 5) if rate > 0 else ""
            st.markdown(
                f"`{row['merchant_category']:<20}` "
                f"<span class='{color}'>{rate:.0f}%</span> {bar} "
                f"({int(row['fraud'])} fraud / {int(row['total'])} total)",
                unsafe_allow_html=True
            )

    with risk_col:
        st.markdown("### Risk Level Distribution")
        risk_counts = df["risk_level"].value_counts()
        st.bar_chart(risk_counts, use_container_width=True, height=250)

        st.markdown("**Financial Impact by Risk:**")
        for risk in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "MINIMAL"]:
            risk_df = df[(df["risk_level"] == risk) & (df["is_fraud"] == 1)]
            if len(risk_df) > 0:
                total_amt = risk_df["amount"].sum()
                color = "critical" if risk == "CRITICAL" else \
                        "high"     if risk == "HIGH"      else \
                        "medium"   if risk == "MEDIUM"    else "low"
                st.markdown(
                    f"<span class='{color}'>{risk}</span>: "
                    f"{len(risk_df)} transactions | ${total_amt:,.0f}",
                    unsafe_allow_html=True
                )

    st.divider()

    # ── ROW 3: User Analysis + Time Heatmap ───────────────────────────────
    user_col, time_col = st.columns([1, 1])

    with user_col:
        st.markdown("### Top Flagged Users")
        user_stats = (
            df[df["is_fraud"] == 1]
            .groupby("user_id")
            .agg(
                fraud_count    = ("is_fraud",          "sum"),
                total_amount   = ("amount",             "sum"),
                avg_prob       = ("fraud_probability",  "mean"),
                last_seen      = ("timestamp",          "last")
            )
            .reset_index()
            .sort_values("fraud_count", ascending=False)
            .head(8)
        )

        if len(user_stats) > 0:
            for _, row in user_stats.iterrows():
                risk_color = "critical" if row["fraud_count"] >= 3 else \
                             "high"     if row["fraud_count"] >= 2 else "medium"
                st.markdown(
                    f"<span class='{risk_color}'>User {int(row['user_id'])}</span> — "
                    f"{int(row['fraud_count'])} fraud | "
                    f"${row['total_amount']:,.0f} | "
                    f"Prob: {row['avg_prob']:.2f} | "
                    f"Last: {row['last_seen']}",
                    unsafe_allow_html=True
                )
        else:
            st.info("No fraud detected yet. Run more scans.")

    with time_col:
        st.markdown("### Fraud by Hour of Day")
        if "hour_of_day" in df.columns:
            hour_stats = df.groupby("hour_of_day").agg(
                total = ("is_fraud", "count"),
                fraud = ("is_fraud", "sum")
            ).reset_index()
            hour_stats["fraud_rate"] = (
                hour_stats["fraud"] / hour_stats["total"] * 100
            ).round(1)
            # Fill missing hours
            all_hours = pd.DataFrame({"hour_of_day": range(24)})
            hour_stats = all_hours.merge(hour_stats, on="hour_of_day", how="left").fillna(0)

            st.bar_chart(
                hour_stats.set_index("hour_of_day")["fraud_rate"],
                use_container_width=True,
                height=250
            )
            st.caption("Fraud rate % by hour — peaks typically 2am-5am")

    st.divider()

    # ── ROW 4: Live Transaction Feed ───────────────────────────────────────
    st.markdown("### Live Transaction Feed")

    feed_col1, feed_col2 = st.columns([3, 1])
    with feed_col1:
        show_all = st.checkbox("Show all transactions (default: fraud only)")
    with feed_col2:
        st.caption(f"Showing last 15 | Total: {len(df)}")

    display_df = df if show_all else df[df["is_fraud"] == 1]
    display_df = display_df.tail(15).iloc[::-1]

    for _, row in display_df.iterrows():
        badge = (
            f'<span class="block-badge">BLOCK</span>'   if row["decision"] == "BLOCK"  else
            f'<span class="review-badge">REVIEW</span>' if row["decision"] == "REVIEW" else
            f'<span class="approve-badge">APPROVE</span>'
        )
        risk_class = row["risk_level"].lower()
        flag       = "🚨" if row["is_fraud"] == 1 else "✅"

        st.markdown(
            f"{flag} `{row['timestamp']}` | "
            f"User **{row['user_id']}** | "
            f"**${row['amount']:,.2f}** | "
            f"`{row['merchant_category']}` | "
            f"Prob: **{row['fraud_probability']:.3f}** | "
            f"<span class='{risk_class}'>{row['risk_level']}</span> | "
            f"{badge} | "
            f"{row['inference_ms']}ms",
            unsafe_allow_html=True
        )

else:
    st.info("Click 'Scan Transactions' to start real-time fraud detection.")
    st.markdown("""
    **What you'll see:**
    - Live fraud probability scoring for every transaction
    - Financial impact tracking ($ blocked, $ at risk)
    - Per-merchant fraud rates
    - Suspicious user profiling
    - Hour-by-hour fraud heatmap
    - Explainable AI — WHY each transaction was flagged
    """)


# ─────────────────────────────────────────────────────────────────────────────
# AUTO REFRESH
# ─────────────────────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(10)
    st.rerun()