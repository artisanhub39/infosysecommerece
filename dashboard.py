# 🛍 Real-Time Competitor Strategy Tracker for E-commerce
# ✅ Uses real CSVs only + includes Colorful Login Page + Fixed rerun

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from transformers import pipeline
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import plotly.express as px

# ----------------- CONFIG -----------------
EMAIL_ADDRESS = "cheemalapoojithareddy955@gmail.com"
EMAIL_PASSWORD = "xnrh bdgu svfr aput"
RECIPIENT_EMAIL = "reddycheemala37@gmail.com"
PRICE_CHANGE_THRESHOLD_PERCENT = 0.05  # 5%

PRICING_CSV = "pricing.csv"
REVIEWS_CSV = "reviews.csv"

# Login credentials
USERNAME = "admin"
PASSWORD = "1234"  # change as you like
# -----------------------------------------

st.set_page_config(layout="wide", page_title="🛍Real-Time Competitor Strategy Tracker")

# ----------------- STYLING -----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1e3c72, #2a5298, #00c6ff);
    color: white;
    font-family: 'Poppins', sans-serif;
}
h1, h2, h3, h4, h5, label, p {
    color: white !important;
}
.login-container {
    background: rgba(255, 255, 255, 0.15);
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0 4px 30px rgba(0,0,0,0.2);
    backdrop-filter: blur(10px);
    width: 400px;
    margin: auto;
    margin-top: 10%;
}
.login-title {
    text-align: center;
    font-size: 1.8rem;
    font-weight: 700;
    color: #fff;
}
input, select, textarea {
    background-color: rgba(255,255,255,0.1) !important;
    color: white !important;
    border-radius: 10px;
}
.stButton>button {
    background: linear-gradient(45deg, #ff6a00, #ee0979);
    color: white;
    font-weight: bold;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1.5rem;
}
.stButton>button:hover {
    background: linear-gradient(45deg, #ee0979, #ff6a00);
}
</style>
""", unsafe_allow_html=True)

# ----------------- LOGIN PAGE -----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<p class="login-title">🔑 Login to Access Dashboard</p>', unsafe_allow_html=True)

    username_input = st.text_input("👤 Username")
    password_input = st.text_input("🔒 Password", type="password")

    login_button = st.button("Login")

    if login_button:
        if username_input == USERNAME and password_input == PASSWORD:
            st.session_state.logged_in = True
            st.success("✅ Welcome admin! Redirecting to dashboard...")
            st.rerun()  # ✅ fixed rerun
        else:
            st.error("❌ Invalid username or password.")

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ----------------- 1) Load CSV Data -----------------
@st.cache_data
def load_csvs(pricing_csv=PRICING_CSV, reviews_csv=REVIEWS_CSV):
    if not os.path.exists(pricing_csv):
        st.error(f"❌ Pricing CSV not found: {pricing_csv}")
        st.stop()
    if not os.path.exists(reviews_csv):
        st.error(f"❌ Reviews CSV not found: {reviews_csv}")
        st.stop()

    df_pricing = pd.read_csv(pricing_csv)
    df_reviews = pd.read_csv(reviews_csv)

    df_pricing["Date"] = pd.to_datetime(df_pricing["Date"], errors="coerce")
    df_reviews["Date"] = pd.to_datetime(df_reviews["Date"], errors="coerce")

    return df_pricing, df_reviews

df_pricing, df_reviews_raw = load_csvs()

# ----------------- 2) Sentiment Model -----------------
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_pipeline = load_sentiment_model()

@st.cache_data
def analyze_sentiment(df_reviews):
    if df_reviews.empty:
        return df_reviews
    texts = df_reviews["Review_Text"].astype(str).tolist()
    results = sentiment_pipeline(texts)
    df = df_reviews.copy()
    df["Predicted_Sentiment"] = [r["label"] for r in results]
    df["Confidence"] = [r["score"] for r in results]
    return df

df_reviews_analyzed = analyze_sentiment(df_reviews_raw)

# ----------------- 3) Email Alerts -----------------
def send_email_alert(sku, price_change_pct, old_price, new_price):
    change_type = "DROP" if price_change_pct < 0 else "INCREASE"
    price_diff = abs(new_price - old_price)
    subject = f"PRICE ALERT: {change_type} for {sku} ({abs(price_change_pct):.2%})"
    html_body = f"""
    <html><body>
      <h3>Market Price Shift Detected for {sku}</h3>
      <p><strong>Change:</strong> {abs(price_change_pct):.2%} ({change_type}) — ₹{price_diff:.2f}</p>
      <p><strong>Old Price:</strong> ₹{old_price:.2f} — <strong>New Price:</strong> ₹{new_price:.2f}</p>
      <p>Generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </body></html>
    """
    msg = MIMEMultipart()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = RECIPIENT_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html"))
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587, timeout=20)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, RECIPIENT_EMAIL, msg.as_string())
        server.quit()
        return True, None
    except Exception as e:
        return False, str(e)

# ----------------- 4) Price Alert System -----------------
def check_and_alert_price_change(df_pricing_data, threshold=PRICE_CHANGE_THRESHOLD_PERCENT):
    alerts = []
    df = df_pricing_data.sort_values(["Product_SKU", "Date"])
    for sku in df["Product_SKU"].unique():
        df_sku = df[df["Product_SKU"] == sku].tail(7)
        if len(df_sku) < 2:
            continue
        new_price = float(df_sku["Our_Price"].iloc[-1])
        old_price = float(df_sku["Our_Price"].iloc[-6:-1].mean()) if len(df_sku) >= 6 else float(df_sku["Our_Price"].iloc[-2])
        if old_price == 0:
            continue
        price_change_pct = (new_price - old_price) / old_price
        if abs(price_change_pct) >= threshold:
            sent, err = send_email_alert(sku, price_change_pct, old_price, new_price)
            alerts.append({
                "SKU": sku,
                "Old_Price": old_price,
                "New_Price": new_price,
                "Change_Pct": price_change_pct,
                "Email_Sent": sent,
                "Error": err
            })
    return alerts

alerts_report = check_and_alert_price_change(df_pricing)

# ----------------- 5) Predictive Model -----------------
@st.cache_data
def predict_next_price(df_pricing_product):
    df = df_pricing_product.sort_values("Date").copy()
    df['Day'] = np.arange(len(df))
    feature_cols = ["Day", "Our_Inventory_Level", "Is_Our_Promo", "Competitor_Price_CompA", "Competitor_Price_CompB"]
    X = df[feature_cols]
    y = df["Our_Price"]
    if len(df) < 3:
        return df["Our_Price"].iloc[-1]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    last_row = df.iloc[-1].copy()
    next_day_features = last_row[feature_cols].values.reshape(1, -1)
    next_day_features[0, 0] = last_row['Day'] + 1
    predicted_price = model.predict(next_day_features)[0]
    return round(predicted_price, 2)

# ----------------- 6) DASHBOARD UI -----------------
st.title("📊 Real-Time Competitor Strategy Tracker for E-commerce")

with st.sidebar:
    st.header("Navigation & Filters")
    page_selection = st.radio("Go to:", ["Product Performance Analysis", "Competitor & Predictive Insights"])
    selected_product_sku = st.selectbox("Select Your Product SKU:", df_pricing["Product_SKU"].unique())
    st.caption("Email alerts are automatically triggered on price changes.")

if alerts_report:
    st.sidebar.success(f"{len(alerts_report)} alert(s) generated this run.")
else:
    st.sidebar.info("No alerts triggered this run.")

df_product_pricing = df_pricing[df_pricing["Product_SKU"] == selected_product_sku].sort_values("Date")
df_product_reviews = df_reviews_analyzed[df_reviews_analyzed["Product_SKU"] == selected_product_sku].sort_values("Date", ascending=False)

# ----------------- PAGE 1: Product Performance -----------------
if page_selection == "Product Performance Analysis":
    st.header(f"📈 Product Performance: {selected_product_sku}")
    
    if not df_product_reviews.empty:
        col1, col2 = st.columns([1, 1])
        
        sentiment_counts = df_product_reviews["Predicted_Sentiment"].value_counts(normalize=True).reset_index()
        sentiment_counts.columns = ["Sentiment", "Proportion"]
        fig = px.pie(
            sentiment_counts,
            names="Sentiment",
            values="Proportion",
            hole=0.4,
            color="Sentiment",
            color_discrete_map={"POSITIVE": "#4CAF50", "NEGATIVE": "#d62728"},
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e6eef8")
        col1.plotly_chart(fig, use_container_width=True)
        
        col2.subheader("📝 Recent Reviews & Predicted Sentiments")
        col2.dataframe(
            df_product_reviews[["Date", "Review_Text", "Rating", "Predicted_Sentiment", "Confidence"]]
            .reset_index(drop=True)
        )
    else:
        st.info("No reviews found for this product.")
    
    st.subheader("Key Metrics")
    total_reviews = len(df_product_reviews)
    avg_rating = df_product_reviews["Rating"].mean() if total_reviews > 0 else np.nan
    pos_reviews = (df_product_reviews["Predicted_Sentiment"] == "POSITIVE").sum() if total_reviews > 0 else 0
    sentiment_score = (pos_reviews / total_reviews * 100) if total_reviews > 0 else 0
    st.metric("Customer Sentiment Score", f"{sentiment_score:.1f}%")
    st.metric("Total Reviews", f"{total_reviews}")
    st.metric("Average Rating", f"{avg_rating:.2f}" if not np.isnan(avg_rating) else "N/A")

# ----------------- PAGE 2: Competitor & Prediction -----------------
else:
    st.header(f"💹 Competitor & Price Insights: {selected_product_sku}")
    
    predicted_price = predict_next_price(df_product_pricing)
    st.metric("Predicted Next Day Price", f"₹{predicted_price:.2f}")

    competitor_cols = [c for c in df_product_pricing.columns if c.startswith("Competitor_Price_")]
    if competitor_cols:
        comp_df = df_product_pricing[["Date", "Our_Price"] + competitor_cols].melt(id_vars=["Date"], var_name="Competitor", value_name="Price")
        comp_df["Competitor"] = comp_df["Competitor"].str.replace("Competitor_Price_", "")
        fig = px.line(comp_df, x="Date", y="Price", color="Competitor", title="Price Trends vs Competitors")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e6eef8")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📊 Recent Competitor Data")
        st.dataframe(df_product_pricing[["Date", "Our_Price"] + competitor_cols].sort_values("Date", ascending=False).head(100))
    else:
        st.info("No competitor price data available.")

# ----------------- ALERTS -----------------
st.subheader("📬 Automatic Email Alerts (this session)")
if alerts_report:
    alerts_df = pd.DataFrame(alerts_report)
    alerts_df["Change_Pct"] = alerts_df["Change_Pct"].map(lambda x: f"{x:.2%}")
    st.table(alerts_df)
else:
    st.info("No alerts triggered this run.")
