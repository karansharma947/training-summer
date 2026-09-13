import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import clone


# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Superstore Analysis & Profit Loss Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------
# STYLING  (clean, card-based, college-project friendly)
# ---------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    .main {
        background-color: #F7F9FC;
        background-image: radial-gradient(circle at 20% 50%, rgba(59,110,245,0.03) 0%, transparent 50%),
                          radial-gradient(circle at 80% 20%, rgba(109,140,247,0.03) 0%, transparent 50%);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ---------- APP HEADER ---------- */
    .app-header {
        padding: 1.6rem 2rem;
        background: linear-gradient(135deg, #1A1F4B 0%, #3B6EF5 50%, #6D8CF7 100%);
        border-radius: 18px;
        color: white;
        margin-bottom: 1.6rem;
        box-shadow: 0 8px 24px rgba(59,110,245,0.3);
        position: relative;
        overflow: hidden;
    }
    .app-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 300px;
        height: 300px;
        background: rgba(255,255,255,0.05);
        border-radius: 50%;
    }
    .app-header h1 {
        margin: 0;
        font-size: 1.7rem;
        font-weight: 700;
    }
    .app-header p {
        margin: 0.3rem 0 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }

    /* ---------- CARDS ---------- */
    .card {
        background: white;
        padding: 1.3rem 1.5rem;
        border-radius: 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04), 0 4px 16px rgba(0,0,0,0.06);
        border: 1px solid #EEF1F6;
        margin-bottom: 1.1rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 36px rgba(59,110,245,0.12), 0 4px 12px rgba(0,0,0,0.06);
        border-color: #C5D1F7;
    }

    /* ---------- GLASS INFO CARD ---------- */
    .info-glass {
        background: linear-gradient(135deg, rgba(59,110,245,0.06) 0%, rgba(109,140,247,0.04) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(59,110,245,0.12);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.1rem;
        transition: all 0.3s ease;
    }
    .info-glass:hover {
        box-shadow: 0 8px 32px rgba(59,110,245,0.1);
        border-color: rgba(59,110,245,0.25);
    }

    /* ---------- STAT BADGE ---------- */
    .stat-badge {
        background: white;
        padding: 1rem 1.3rem;
        border-radius: 12px;
        border-left: 4px solid #3B6EF5;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05);
        margin-bottom: 0.8rem;
        transition: all 0.3s ease;
    }
    .stat-badge:hover {
        transform: translateX(4px);
        box-shadow: 0 6px 20px rgba(59,110,245,0.12);
    }
    .stat-badge.green  { border-left-color: #1FAA59; }
    .stat-badge.red    { border-left-color: #E23744; }
    .stat-badge.purple { border-left-color: #9B5EF5; }
    .stat-badge.orange { border-left-color: #F59B3B; }
    .stat-badge .stat-label {
        font-size: 0.78rem;
        color: #7A8399;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-weight: 500;
        margin: 0;
    }
    .stat-badge .stat-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1E2749;
        margin: 0.15rem 0 0 0;
    }

    /* ---------- METRIC CARDS ---------- */
    .metric-card {
        background: white;
        padding: 1rem 1.2rem;
        border-radius: 14px;
        border: 1px solid #EEF1F6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04), 0 4px 14px rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #3B6EF5, #6D8CF7, #9B5EF5);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 12px 32px rgba(59,110,245,0.15), 0 4px 10px rgba(0,0,0,0.05);
        border-color: #8FA0FF;
    }
    .metric-card:hover::after {
        opacity: 1;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #7A8399;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .metric-card .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1E2749;
    }
    .metric-card .icon {
        font-size: 1.6rem;
        margin-bottom: 0.3rem;
    }

    /* ---------- RESULT BADGES ---------- */
    .result-profit {
        background: linear-gradient(135deg, #0D8B44 0%, #1FAA59 40%, #35C77E 100%);
        color: white;
        padding: 2rem;
        border-radius: 18px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(31,170,89,0.3), 0 4px 12px rgba(31,170,89,0.15);
        animation: fadeSlideIn 0.5s ease;
        position: relative;
        overflow: hidden;
    }
    .result-profit::before {
        content: '';
        position: absolute;
        top: -40%; right: -30%;
        width: 200px; height: 200px;
        background: rgba(255,255,255,0.08);
        border-radius: 50%;
    }
    .result-loss {
        background: linear-gradient(135deg, #B91C2E 0%, #E23744 40%, #F1585F 100%);
        color: white;
        padding: 2rem;
        border-radius: 18px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(226,55,68,0.3), 0 4px 12px rgba(226,55,68,0.15);
        animation: fadeSlideIn 0.5s ease;
        position: relative;
        overflow: hidden;
    }
    .result-loss::before {
        content: '';
        position: absolute;
        top: -40%; right: -30%;
        width: 200px; height: 200px;
        background: rgba(255,255,255,0.08);
        border-radius: 50%;
    }
    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .result-profit h2, .result-loss h2 {
        margin: 0;
        font-size: 2.2rem;
        position: relative;
    }
    .result-profit p, .result-loss p {
        margin: 0.3rem 0 0 0;
        opacity: 0.95;
        position: relative;
    }

    /* ---------- PULSING DOT ---------- */
    .pulse-dot {
        display: inline-block;
        width: 10px; height: 10px;
        border-radius: 50%;
        background: #1FAA59;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(31,170,89,0.5); }
        70%  { box-shadow: 0 0 0 8px rgba(31,170,89,0); }
        100% { box-shadow: 0 0 0 0 rgba(31,170,89,0); }
    }

    /* ---------- CONFIDENCE BAR ---------- */
    .confidence-bar-bg {
        background: rgba(255,255,255,0.2);
        border-radius: 20px;
        height: 12px;
        margin: 0.8rem auto 0 auto;
        max-width: 280px;
        overflow: hidden;
        position: relative;
    }
    .confidence-bar-fill {
        height: 100%;
        border-radius: 20px;
        background: rgba(255,255,255,0.85);
        transition: width 1s ease;
    }

    /* ---------- SIDEBAR ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #141937 0%, #1E2749 40%, #253060 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #F7F9FC !important;
    }

    /* Sidebar brand area */
    .sidebar-brand {
        text-align: center;
        padding: 0.6rem 0 0.2rem 0;
    }
    .sidebar-brand h2 {
        margin: 0;
        font-size: 1.45rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        background: linear-gradient(90deg, #8FA0FF, #E0E6FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sidebar-brand p {
        margin: 0.15rem 0 0 0;
        font-size: 0.78rem;
        color: #8FA0FF !important;
        letter-spacing: 0.04em;
        font-weight: 500;
    }

    /* Sidebar navigation buttons */
    .sidebar-nav-btn {
        display: block;
        width: 100%;
        padding: 0.7rem 1rem;
        margin: 4px 0;
        border: none;
        border-radius: 10px;
        text-align: left;
        font-size: 0.9rem;
        font-weight: 500;
        font-family: 'Poppins', sans-serif;
        cursor: pointer;
        transition: all 0.25s ease;
        background: rgba(255,255,255,0.06);
        color: #CBD2E8 !important;
    }
    .sidebar-nav-btn:hover {
        background: rgba(59,110,245,0.35);
        color: #FFFFFF !important;
        transform: translateX(4px);
        box-shadow: 0 2px 10px rgba(59,110,245,0.2);
    }
    .sidebar-nav-btn.active {
        background: linear-gradient(120deg, #3B6EF5 0%, #5A80F9 100%);
        color: #FFFFFF !important;
        font-weight: 600;
        box-shadow: 0 4px 14px rgba(59,110,245,0.4);
    }
    .nav-icon {
        margin-right: 0.5rem;
        font-size: 1rem;
    }

    /* ---------- PRIMARY ACTION BUTTON ---------- */
    .stButton>button {
        background: linear-gradient(135deg, #3B6EF5 0%, #6D8CF7 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.6rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px rgba(59,110,245,0.25);
        position: relative;
        overflow: hidden;
    }
    .stButton>button::after {
        content: '';
        position: absolute;
        top: 0; left: -100%; width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(59,110,245,0.4);
        color: white;
    }
    .stButton>button:hover::after {
        left: 100%;
    }
    .stButton>button:active {
        transform: translateY(0);
        box-shadow: 0 2px 8px rgba(59,110,245,0.25);
    }

    /* ---------- DOWNLOAD BUTTON ---------- */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #6D8CF7 0%, #9B5EF5 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(155,94,245,0.2);
    }
    .stDownloadButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(155,94,245,0.35);
        color: white;
    }

    /* ---------- INPUT WIDGETS ---------- */
    .stNumberInput, .stSelectbox, .stSlider, .stMultiselect {
        transition: all 0.2s ease;
    }

    /* ---------- SECTION TITLE ---------- */
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1E2749;
        margin-bottom: 0.6rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid transparent;
        border-image: linear-gradient(90deg, #3B6EF5, transparent) 1;
    }

    /* ---------- DIVIDER ---------- */
    .sidebar-divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.1);
        margin: 0.8rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("superstore_cleaned.csv")
    return df

df = load_data()

FEATURES = [
    "Sales", "Quantity", "Discount", "Delivery Days", "Order Month Number",
    "Ship Mode", "Segment", "Region", "Category", "Sub-Category"
]
CATEGORICAL = ["Ship Mode", "Segment", "Region", "Category", "Sub-Category"]
NUMERIC = ["Sales", "Quantity", "Discount", "Delivery Days", "Order Month Number"]

# ---------------------------------------------------------------
# MODEL TRAINING (cached so it only runs once per session)
# ---------------------------------------------------------------
@st.cache_resource
def train_models(data: pd.DataFrame):
    if os.path.exists("superstore_profit_loss_model.pkl"):
        try:
            with open("superstore_profit_loss_model.pkl", "rb") as f:
                bundle = pickle.load(f)
                return bundle
        except Exception:
            pass

    X = data[FEATURES]
    y = data["Profit Status"]
    X_encoded = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    log_model = LogisticRegression(max_iter=500)
    log_model.fit(X_train_scaled, y_train)
    y_pred = log_model.predict(X_test_scaled)
    results["Logistic Regression"] = {
        "model": log_model, "uses_scaling": True,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="Loss", zero_division=0),
        "recall": recall_score(y_test, y_pred, pos_label="Loss", zero_division=0),
        "f1": f1_score(y_test, y_pred, pos_label="Loss", zero_division=0),
    }

    dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
    dt_model.fit(X_train, y_train)
    y_pred = dt_model.predict(X_test)
    results["Decision Tree"] = {
        "model": dt_model, "uses_scaling": False,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="Loss", zero_division=0),
        "recall": recall_score(y_test, y_pred, pos_label="Loss", zero_division=0),
        "f1": f1_score(y_test, y_pred, pos_label="Loss", zero_division=0),
    }

    rf_model = RandomForestClassifier(random_state=42, n_estimators=60, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    results["Random Forest"] = {
        "model": rf_model, "uses_scaling": False,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="Loss", zero_division=0),
        "recall": recall_score(y_test, y_pred, pos_label="Loss", zero_division=0),
        "f1": f1_score(y_test, y_pred, pos_label="Loss", zero_division=0),
    }

    comparison = pd.DataFrame([
        {"Model": name, "Accuracy": r["accuracy"], "Loss Precision": r["precision"],
         "Loss Recall": r["recall"], "Loss F1-Score": r["f1"]}
        for name, r in results.items()
    ]).sort_values(by=["Loss F1-Score", "Loss Recall", "Accuracy"], ascending=False).reset_index(drop=True)

    best_name = comparison.iloc[0]["Model"]
    best_info = results[best_name]

    final_model = clone(best_info["model"])
    if hasattr(final_model, "n_jobs"):
        final_model.set_params(n_jobs=-1)

    if best_info["uses_scaling"]:
        final_scaler = StandardScaler()
        X_full = final_scaler.fit_transform(X_encoded)
    else:
        final_scaler = None
        X_full = X_encoded
    final_model.fit(X_full, y)

    return {
        "comparison": comparison,
        "best_name": best_name,
        "final_model": final_model,
        "final_scaler": final_scaler,
        "encoded_columns": X_encoded.columns.tolist(),
    }

model_bundle = train_models(df)

def predict_profit_status(input_dict):
    row = pd.DataFrame([input_dict])
    row_encoded = pd.get_dummies(row, drop_first=True)
    row_encoded = row_encoded.reindex(columns=model_bundle["encoded_columns"], fill_value=0)
    if model_bundle["final_scaler"] is not None:
        row_final = model_bundle["final_scaler"].transform(row_encoded)
    else:
        row_final = row_encoded
    pred = model_bundle["final_model"].predict(row_final)[0]
    proba = None
    if hasattr(model_bundle["final_model"], "predict_proba"):
        classes = list(model_bundle["final_model"].classes_)
        probs = model_bundle["final_model"].predict_proba(row_final)[0]
        proba = dict(zip(classes, probs))
    return pred, proba

# ---------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------
# Navigation pages config
NAV_PAGES = [
    {"label": "Profit / Loss Predictor", "icon": "🔮", "key": "Profit/Loss Predictor"},
    {"label": "Data View",              "icon": "📋", "key": "Data View"},
    {"label": "Visualizations",         "icon": "📈", "key": "Visualizations"},
    {"label": "Findings & Conclusion",  "icon": "📝", "key": "Findings & Conclusion"},
]

if "active_page" not in st.session_state:
    st.session_state.active_page = "Profit/Loss Predictor"

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h2>Superstore</h2>
        <p>Analysis &amp; Profit Loss Predictor</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # Navigation buttons
    for nav in NAV_PAGES:
        is_active = st.session_state.active_page == nav["key"]
        active_class = "active" if is_active else ""
        btn_clicked = st.button(
            f"{nav['icon']}  {nav['label']}",
            key=f"nav_{nav['key']}",
            use_container_width=True,
        )
        if btn_clicked:
            st.session_state.active_page = nav["key"]
            st.rerun()

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.caption(f"Dataset: {df.shape[0]:,} orders · {df.shape[1]} columns")

page = st.session_state.active_page

# =================================================================
# PAGE 1 — PROFIT/LOSS PREDICTOR (HOME)
# =================================================================
if page == "Profit/Loss Predictor":
    st.markdown("""
    <div class="app-header">
        <h1>📊 Superstore Profit / Loss Predictor</h1>
        <p>Enter order details below to predict whether the order will result in a Profit or a Loss.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Quick Stats Row ---
    loss_count = int((df["Profit Status"] == "Loss").sum())
    profit_count = int((df["Profit Status"] == "Profit").sum())
    avg_discount = df["Discount"].mean()
    avg_profit = df["Profit"].mean()

    qs1, qs2, qs3, qs4 = st.columns(4)
    for col_qs, icon_qs, label_qs, value_qs in zip(
        [qs1, qs2, qs3, qs4],
        ["📦", "✅", "⚠️", "💰"],
        ["Total Orders", "Profitable", "Loss-Making", "Avg Profit"],
        [f"{len(df):,}", f"{profit_count:,}", f"{loss_count:,}", f"${avg_profit:,.2f}"],
    ):
        col_qs.markdown(f"""<div class="metric-card">
            <div class="icon">{icon_qs}</div>
            <div class="label">{label_qs}</div>
            <div class="value">{value_qs}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        st.markdown('<div class="section-title">🛒 Order Details</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            sales = st.number_input("Sales ($)", min_value=0.0, value=100.0, step=10.0)
            quantity = st.number_input("Quantity", min_value=1, max_value=20, value=2, step=1)
            discount = st.slider("Discount", min_value=0.0, max_value=0.8, value=0.0, step=0.05)
        with c2:
            delivery_days = st.number_input("Delivery Days", min_value=0, max_value=14, value=4, step=1)
            month = st.selectbox("Order Month", list(range(1, 13)), index=10,
                                  format_func=lambda m: pd.Timestamp(2020, m, 1).strftime("%B"))

        c3, c4 = st.columns(2)
        with c3:
            ship_mode = st.selectbox("Ship Mode", sorted(df["Ship Mode"].unique()))
            segment = st.selectbox("Segment", sorted(df["Segment"].unique()))
        with c4:
            region = st.selectbox("Region", sorted(df["Region"].unique()))
            category = st.selectbox("Category", sorted(df["Category"].unique()))

        sub_options = sorted(df[df["Category"] == category]["Sub-Category"].unique())
        sub_category = st.selectbox("Sub-Category", sub_options)

        predict_btn = st.button("🔮 Predict Profit / Loss")


        # --- Discount Risk Indicator (always visible) ---
        if discount >= 0.3:
            risk_color, risk_label, risk_icon = "#E23744", "High Risk", "🔴"
        elif discount >= 0.15:
            risk_color, risk_label, risk_icon = "#F59B3B", "Moderate Risk", "🟡"
        else:
            risk_color, risk_label, risk_icon = "#1FAA59", "Low Risk", "🟢"

        high_disc_loss = df[df["Discount"] >= 0.3]
        high_disc_rate = (high_disc_loss["Profit Status"] == "Loss").mean() * 100 if len(high_disc_loss) else 0

        st.markdown(f"""
        <div class="info-glass">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:0.5rem;">
                <span style="font-size:1.3rem;">{risk_icon}</span>
                <span style="font-weight:600; color:#1E2749; font-size:1rem;">Discount Risk: <span style="color:{risk_color}">{risk_label}</span></span>
            </div>
            <p style="margin:0; color:#5A6378; font-size:0.85rem;">
                Your discount is set to <b>{discount*100:.0f}%</b>. In the dataset, orders with ≥30% discount have a
                <b style="color:#E23744">{high_disc_rate:.1f}%</b> loss rate.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        if predict_btn:
            input_dict = {
                "Sales": sales, "Quantity": quantity, "Discount": discount,
                "Delivery Days": delivery_days, "Order Month Number": month,
                "Ship Mode": ship_mode, "Segment": segment, "Region": region,
                "Category": category, "Sub-Category": sub_category,
            }
            pred, proba = predict_profit_status(input_dict)

            css_class = "result-profit" if pred == "Profit" else "result-loss"
            icon = "✅" if pred == "Profit" else "⚠️"
            confidence = proba[pred] * 100 if proba else None

            conf_bar_html = ""
            if confidence:
                conf_bar_html = f'<div class="confidence-bar-bg"><div class="confidence-bar-fill" style="width:{confidence:.0f}%"></div></div>'

            st.markdown(f"""
            <div class="{css_class}">
                <h2>{icon} Predicted: {pred}</h2>
                <p>{'This order is expected to be profitable.' if pred=='Profit' else 'This order is expected to result in a loss.'}</p>
                {f'<p style="font-size:1.5rem;font-weight:700;margin-top:0.6rem;position:relative;">{confidence:.1f}% confidence</p>' if confidence else ''}
                {conf_bar_html}
            </div>
            """, unsafe_allow_html=True)

            if proba:
                st.markdown("<br>", unsafe_allow_html=True)

                # Confidence gauge donut
                profit_prob = proba.get("Profit", 0)
                loss_prob = proba.get("Loss", 0)
                fig_gauge = go.Figure(go.Pie(
                    values=[profit_prob, loss_prob],
                    labels=["Profit", "Loss"],
                    hole=0.7,
                    marker=dict(colors=["#1FAA59", "#E23744"]),
                    textinfo="label+percent",
                    textfont=dict(size=13),
                    hoverinfo="label+percent",
                ))
                fig_gauge.update_layout(
                    height=220, margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    annotations=[dict(text=f"<b>{confidence:.0f}%</b>" if confidence else "",
                                       x=0.5, y=0.5, font_size=22, showarrow=False,
                                       font_color="#1E2749")]
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Probability bar
                fig = go.Figure(go.Bar(
                    x=list(proba.values()), y=list(proba.keys()), orientation="h",
                    marker_color=["#1FAA59" if k == "Profit" else "#E23744" for k in proba.keys()],
                    text=[f"{v*100:.1f}%" for v in proba.values()], textposition="outside",
                    marker=dict(line=dict(width=0), cornerradius=6),
                ))
                fig.update_layout(
                    height=140, margin=dict(l=10, r=40, t=10, b=10),
                    xaxis=dict(range=[0, 1], title="Probability", showgrid=False),
                    yaxis=dict(showgrid=False),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

            # Input summary badges
            st.markdown(f"""
            <div class="stat-badge green">
                <p class="stat-label">Order Summary</p>
                <p class="stat-value">${sales:,.2f} × {quantity} items</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="stat-badge {'red' if discount >= 0.3 else 'orange' if discount >= 0.15 else 'purple'}">
                <p class="stat-label">Discount Applied</p>
                <p class="stat-value">{discount*100:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="card" style="text-align:center; padding:3rem 1.5rem;">
                <div style="font-size:3rem; margin-bottom:0.5rem;">🔮</div>
                <h3 style="color:#3B6EF5; margin:0;">Ready to Predict</h3>
                <p style="color:#7A8399; margin:0.5rem 0 0 0;">Fill in the order details on the left and click <b>Predict</b> to see the result</p>
            </div>
            """, unsafe_allow_html=True)

            # Quick insights while waiting
            cat_profit = df.groupby("Category")["Profit"].sum().sort_values(ascending=True)
            fig_cat = px.bar(
                x=cat_profit.values, y=cat_profit.index, orientation="h",
                color=cat_profit.values,
                color_continuous_scale=["#E23744", "#F59B3B", "#1FAA59"],
                labels={"x": "Total Profit ($)", "y": ""},
            )
            fig_cat.update_layout(
                height=200, margin=dict(l=10, r=10, t=30, b=10),
                title=dict(text="Profit by Category", font=dict(size=14, color="#1E2749")),
                showlegend=False, coloraxis_showscale=False,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_cat, use_container_width=True)

        # Model in Use section with pulse dot
        best_acc = model_bundle["comparison"].iloc[0]["Accuracy"]
        st.markdown(f"""
        <div class="card">
            <div class="section-title"><span class="pulse-dot"></span> Model in Use</div>
            <p style="color:#555; margin:0;">Best-performing model selected automatically: <b style="color:#3B6EF5">{model_bundle['best_name']}</b></p>
            <p style="color:#7A8399; font-size:0.85rem; margin:0.3rem 0 0 0;">Ranked by Loss F1-Score → Loss Recall → Accuracy &nbsp;|&nbsp; Accuracy: <b>{best_acc*100:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)

    # --- Bottom row: Dataset quick insights ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Quick Dataset Insights</div>', unsafe_allow_html=True)

    ins1, ins2, ins3 = st.columns(3)

    with ins1:
        region_profit = df.groupby("Region")["Profit"].sum().reset_index().sort_values("Profit")
        fig_r = px.bar(region_profit, x="Profit", y="Region", orientation="h",
                       color="Profit", color_continuous_scale=["#E23744", "#35C77E"])
        fig_r.update_layout(height=250, margin=dict(l=10, r=10, t=35, b=10),
                            title=dict(text="Profit by Region", font=dict(size=13, color="#1E2749")),
                            coloraxis_showscale=False,
                            paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig_r, use_container_width=True)

    with ins2:
        seg_data = df.groupby("Segment")["Profit Status"].value_counts().unstack(fill_value=0)
        fig_s = go.Figure()
        if "Profit" in seg_data.columns:
            fig_s.add_trace(go.Bar(name="Profit", x=seg_data.index, y=seg_data["Profit"],
                                    marker_color="#1FAA59"))
        if "Loss" in seg_data.columns:
            fig_s.add_trace(go.Bar(name="Loss", x=seg_data.index, y=seg_data["Loss"],
                                    marker_color="#E23744"))
        fig_s.update_layout(barmode="group", height=250, margin=dict(l=10, r=10, t=35, b=10),
                            title=dict(text="Profit vs Loss by Segment", font=dict(size=13, color="#1E2749")),
                            legend=dict(orientation="h", y=1.15),
                            paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig_s, use_container_width=True)

    with ins3:
        ship_loss = df.groupby("Ship Mode")["Profit Status"].apply(
            lambda s: (s == "Loss").mean() * 100
        ).reset_index(name="Loss Rate (%)")
        fig_sh = px.bar(ship_loss, x="Ship Mode", y="Loss Rate (%)",
                        color="Loss Rate (%)", color_continuous_scale=["#35C77E", "#E23744"])
        fig_sh.update_layout(height=250, margin=dict(l=10, r=10, t=35, b=10),
                             title=dict(text="Loss Rate by Ship Mode", font=dict(size=13, color="#1E2749")),
                             coloraxis_showscale=False,
                             paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig_sh, use_container_width=True)

# =================================================================
# PAGE 2 — DATA VIEW
# =================================================================
elif page == "Data View":
    st.markdown("""
    <div class="app-header">
        <h1>🗂️ Dataset Explorer</h1>
        <p>Browse, filter, and download the cleaned Superstore dataset.</p>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    for col, icon_m, label, value in zip(
        [m1, m2, m3, m4],
        ["📦", "💵", "📈", "📉"],
        ["Total Orders", "Total Sales", "Total Profit", "Loss Orders"],
        [f"{len(df):,}", f"${df['Sales'].sum():,.0f}", f"${df['Profit'].sum():,.0f}",
         f"{(df['Profit Status']=='Loss').sum():,} ({(df['Profit Status']=='Loss').mean()*100:.1f}%)"]
    ):
        col.markdown(f"""<div class="metric-card">
            <div class="icon">{icon_m}</div>
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔍 Filters</div>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        region_f = st.multiselect("Region", sorted(df["Region"].unique()))
    with f2:
        category_f = st.multiselect("Category", sorted(df["Category"].unique()))
    with f3:
        segment_f = st.multiselect("Segment", sorted(df["Segment"].unique()))
    with f4:
        status_f = st.multiselect("Profit Status", sorted(df["Profit Status"].unique()))

    filtered = df.copy()
    if region_f:
        filtered = filtered[filtered["Region"].isin(region_f)]
    if category_f:
        filtered = filtered[filtered["Category"].isin(category_f)]
    if segment_f:
        filtered = filtered[filtered["Segment"].isin(segment_f)]
    if status_f:
        filtered = filtered[filtered["Profit Status"].isin(status_f)]



    st.markdown(f'<div class="section-title">📋 Showing {len(filtered):,} of {len(df):,} records</div>',
                unsafe_allow_html=True)
    st.dataframe(filtered, use_container_width=True, height=430)

    st.download_button(
        "⬇️ Download Filtered Data (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="superstore_filtered.csv",
        mime="text/csv",
    )

    # --- Filtered data visual summaries ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Filtered Data Overview</div>', unsafe_allow_html=True)

    dv1, dv2 = st.columns(2)
    with dv1:

        filt_status = filtered["Profit Status"].value_counts().reset_index()
        filt_status.columns = ["Profit Status", "Count"]
        fig_dv1 = px.pie(filt_status, names="Profit Status", values="Count", hole=0.55,
                         color="Profit Status", color_discrete_map={"Profit": "#1FAA59", "Loss": "#E23744"})
        fig_dv1.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10),
                              title=dict(text="Profit vs Loss Split", font=dict(size=13, color="#1E2749")),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_dv1, use_container_width=True)


    with dv2:

        fig_dv2 = px.histogram(filtered, x="Sales", nbins=40, color_discrete_sequence=["#3B6EF5"],
                               opacity=0.8)
        fig_dv2.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10),
                              title=dict(text="Sales Distribution", font=dict(size=13, color="#1E2749")),
                              xaxis_title="Sales ($)", yaxis_title="Count",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_dv2, use_container_width=True)


# =================================================================
# PAGE 3 — VISUALIZATIONS
# =================================================================
elif page == "Visualizations":
    st.markdown("""
    <div class="app-header">
        <h1>📈 Visual Insights</h1>
        <p>Explore sales, profit, and discount patterns across the business.</p>
    </div>
    """, unsafe_allow_html=True)

    row1c1, row1c2 = st.columns(2)
    with row1c1:
        st.markdown('<div class="section-title">Profit Status Distribution</div>', unsafe_allow_html=True)
        pie_data = df["Profit Status"].value_counts().reset_index()
        pie_data.columns = ["Profit Status", "Count"]
        fig = px.pie(pie_data, names="Profit Status", values="Count", hole=0.5,
                     color="Profit Status", color_discrete_map={"Profit": "#1FAA59", "Loss": "#E23744"})
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
        st.plotly_chart(fig, use_container_width=True)


    with row1c2:
        st.markdown('<div class="section-title">Sales vs Profit by Category</div>', unsafe_allow_html=True)
        cat_agg = df.groupby("Category")[["Sales", "Profit"]].sum().reset_index()
        fig = px.bar(cat_agg, x="Category", y=["Sales", "Profit"], barmode="group",
                     color_discrete_sequence=["#3B6EF5", "#35C77E"])
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320,
                           legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, use_container_width=True)


    row2c1, row2c2 = st.columns(2)
    with row2c1:
        st.markdown('<div class="section-title">Profit by Region</div>', unsafe_allow_html=True)
        reg_agg = df.groupby("Region")["Profit"].sum().reset_index().sort_values("Profit")
        fig = px.bar(reg_agg, x="Profit", y="Region", orientation="h",
                     color="Profit", color_continuous_scale=["#E23744", "#35C77E"])
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
        st.plotly_chart(fig, use_container_width=True)


    with row2c2:
        st.markdown('<div class="section-title">Discount vs Profit</div>', unsafe_allow_html=True)
        sample = df.sample(min(1500, len(df)), random_state=1)
        fig = px.scatter(sample, x="Discount", y="Profit", color="Profit Status",
                          color_discrete_map={"Profit": "#1FAA59", "Loss": "#E23744"}, opacity=0.6)
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
        st.plotly_chart(fig, use_container_width=True)


    st.markdown('<div class="section-title">Monthly Sales & Profit Trend</div>', unsafe_allow_html=True)
    month_agg = df.groupby(["Order Year", "Order Month Number"])[["Sales", "Profit"]].sum().reset_index()
    month_agg["Period"] = month_agg["Order Year"].astype(str) + "-" + month_agg["Order Month Number"].astype(str).str.zfill(2)
    month_agg = month_agg.sort_values(["Order Year", "Order Month Number"])
    fig = px.line(month_agg, x="Period", y=["Sales", "Profit"],
                  color_discrete_sequence=["#3B6EF5", "#35C77E"])
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=340,
                       legend=dict(orientation="h", y=1.15), xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


    st.markdown('<div class="section-title">Loss Rate by Sub-Category</div>', unsafe_allow_html=True)
    loss_rate = df.groupby("Sub-Category")["Profit Status"].apply(
        lambda s: (s == "Loss").mean() * 100
    ).reset_index(name="Loss Rate (%)").sort_values("Loss Rate (%)", ascending=False)
    fig = px.bar(loss_rate, x="Sub-Category", y="Loss Rate (%)",
                 color="Loss Rate (%)", color_continuous_scale=["#35C77E", "#E23744"])
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=340, xaxis_tickangle=-30,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)



    # --- Top vs Bottom Sub-Categories by Profit ---
    row3c1, row3c2 = st.columns(2)
    with row3c1:
        st.markdown('<div class="section-title">🏆 Top 5 Most Profitable Sub-Categories</div>', unsafe_allow_html=True)
        sub_profit = df.groupby("Sub-Category")["Profit"].sum().sort_values(ascending=False)
        top5 = sub_profit.head(5).reset_index()
        top5.columns = ["Sub-Category", "Profit"]
        fig_top = px.bar(top5, x="Profit", y="Sub-Category", orientation="h",
                         color_discrete_sequence=["#1FAA59"])
        fig_top.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                              yaxis=dict(autorange="reversed"),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_top, use_container_width=True)


    with row3c2:
        st.markdown('<div class="section-title">📉 Top 5 Least Profitable Sub-Categories</div>', unsafe_allow_html=True)
        bot5 = sub_profit.tail(5).reset_index()
        bot5.columns = ["Sub-Category", "Profit"]
        fig_bot = px.bar(bot5, x="Profit", y="Sub-Category", orientation="h",
                         color_discrete_sequence=["#E23744"])
        fig_bot.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bot, use_container_width=True)


# =================================================================
# PAGE 4 — FINDINGS & CONCLUSION
# =================================================================
elif page == "Findings & Conclusion":
    st.markdown("""
    <div class="app-header">
        <h1>📝 Findings & Conclusion</h1>
        <p>Model performance comparison and key business takeaways.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🤖 Model Comparison</div>', unsafe_allow_html=True)
    comp_display = model_bundle["comparison"].copy()
    for c in ["Accuracy", "Loss Precision", "Loss Recall", "Loss F1-Score"]:
        comp_display[c] = (comp_display[c] * 100).round(2).astype(str) + "%"
    st.dataframe(comp_display, use_container_width=True, hide_index=True)
    st.success(f"🏆 Best model selected: **{model_bundle['best_name']}** "
               f"(ranked by Loss F1-Score → Loss Recall → Accuracy)")


    # --- Visual model comparison chart ---
    st.markdown('<div class="section-title">📊 Model Performance Comparison</div>', unsafe_allow_html=True)
    comp_raw = model_bundle["comparison"].copy()
    fig_models = go.Figure()
    colors = ["#3B6EF5", "#1FAA59", "#F59B3B", "#9B5EF5"]
    for i, metric in enumerate(["Accuracy", "Loss Precision", "Loss Recall", "Loss F1-Score"]):
        fig_models.add_trace(go.Bar(
            name=metric, x=comp_raw["Model"], y=comp_raw[metric] * 100,
            marker_color=colors[i],
            text=[f"{v:.1f}%" for v in comp_raw[metric] * 100],
            textposition="outside",
        ))
    fig_models.update_layout(
        barmode="group", height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="Score (%)", range=[0, 105]),
        legend=dict(orientation="h", y=1.15),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_models, use_container_width=True)


    # ---- computed insights ----
    loss_rate_overall = (df["Profit Status"] == "Loss").mean() * 100
    sub_loss = df.groupby("Sub-Category")["Profit Status"].apply(lambda s: (s == "Loss").mean() * 100)
    worst_sub = sub_loss.idxmax()
    worst_sub_rate = sub_loss.max()
    best_region = df.groupby("Region")["Profit"].sum().idxmax()
    worst_region = df.groupby("Region")["Profit"].sum().idxmin()
    high_disc = df[df["Discount"] >= 0.3]
    high_disc_loss_rate = (high_disc["Profit Status"] == "Loss").mean() * 100 if len(high_disc) else 0
    low_disc = df[df["Discount"] < 0.3]
    low_disc_loss_rate = (low_disc["Profit Status"] == "Loss").mean() * 100 if len(low_disc) else 0
    best_category = df.groupby("Category")["Profit"].sum().idxmax()

    # --- Key Findings with stat badges ---
    st.markdown('<div class="section-title">🔍 Key Findings</div>', unsafe_allow_html=True)

    kf1, kf2, kf3, kf4 = st.columns(4)
    kf1.markdown(f"""<div class="stat-badge red">
        <p class="stat-label">Overall Loss Rate</p>
        <p class="stat-value">{loss_rate_overall:.1f}%</p>
    </div>""", unsafe_allow_html=True)
    kf2.markdown(f"""<div class="stat-badge orange">
        <p class="stat-label">Worst Sub-Category</p>
        <p class="stat-value">{worst_sub}</p>
    </div>""", unsafe_allow_html=True)
    kf3.markdown(f"""<div class="stat-badge green">
        <p class="stat-label">Best Region</p>
        <p class="stat-value">{best_region}</p>
    </div>""", unsafe_allow_html=True)
    kf4.markdown(f"""<div class="stat-badge purple">
        <p class="stat-label">Top Category</p>
        <p class="stat-value">{best_category}</p>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    - Overall, **{loss_rate_overall:.1f}%** of all orders in the dataset result in a loss.
    - **{worst_sub}** has the highest loss rate among sub-categories, at **{worst_sub_rate:.1f}%** of its orders.
    - The **{best_region}** region generates the highest total profit, while the **{worst_region}** region generates the lowest.
    - Orders with a discount of **30% or higher** have a loss rate of **{high_disc_loss_rate:.1f}%**, compared to **{low_disc_loss_rate:.1f}%** for orders with lower discounts — deep discounting is strongly linked to losses.
    - The **{best_category}** category contributes the most to overall profit.
    """)


    # --- Discount Impact Visual ---
    disc1, disc2 = st.columns(2)
    with disc1:
        st.markdown('<div class="section-title">⚡ Discount Impact on Loss Rate</div>', unsafe_allow_html=True)
        disc_compare = pd.DataFrame({
            "Discount Level": ["< 30%", "≥ 30%"],
            "Loss Rate (%)": [low_disc_loss_rate, high_disc_loss_rate]
        })
        fig_disc = px.bar(disc_compare, x="Discount Level", y="Loss Rate (%)",
                          color="Discount Level",
                          color_discrete_map={"< 30%": "#1FAA59", "≥ 30%": "#E23744"},
                          text="Loss Rate (%)")
        fig_disc.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_disc.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                               showlegend=False,
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_disc, use_container_width=True)


    with disc2:
        st.markdown('<div class="section-title">🌍 Regional Profit Comparison</div>', unsafe_allow_html=True)
        reg_profit = df.groupby("Region")["Profit"].sum().reset_index().sort_values("Profit", ascending=False)
        fig_reg = px.bar(reg_profit, x="Region", y="Profit",
                         color="Profit", color_continuous_scale=["#E23744", "#35C77E"],
                         text="Profit")
        fig_reg.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig_reg.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                              coloraxis_showscale=False,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_reg, use_container_width=True)


    st.markdown('<div class="section-title">📌 Conclusion</div>', unsafe_allow_html=True)
    st.markdown(f"""
    The analysis shows that **discount level** is one of the strongest drivers of order-level losses —
    orders discounted at 30% or more are considerably more likely to lose money than lightly discounted orders.
    Region and sub-category also matter: certain sub-categories and regions consistently underperform on profit.

    Among the three models tested, **{model_bundle['best_name']}** was selected as the production model because it
    gives the best balance of correctly catching loss-making orders (recall) while keeping overall accuracy high.
    This lets the business flag likely loss-making orders (e.g. before approving a large discount) and intervene early.
    """)

