import streamlit as st
import os
import pandas as pd
from datetime import datetime

# ===== Import EVERYTHING from your existing script =====
from Smart_AI_Tourism_Recommender import (
    DataGenerator,
    MLModels,
    predict_demand_for,
    content_recommendations,
    cf_recommend_for,
    predict_review_sentiment,
    HISTORY_CSV, DEST_CSV, RATINGS_CSV, REVIEWS_CSV, WEATHER_CSV
)

# ============  PAGE CONFIG  ============
st.set_page_config(
    page_title="AI Smart Tourism System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============  CSS FOR CUSTOM DESIGN  ============
st.markdown("""
<style>
    .title {
        font-size: 42px;
        font-weight: 900;
        color: #ffffff;
        text-align: center;
        padding: 15px;
        border-radius: 15px;
        background: linear-gradient(90deg, #6a11cb, #2575fc);
        margin-bottom: 25px;
    }
    
    .section-head {
        font-size: 30px;
        font-weight: 700;
        color: #333;
        margin-top: 30px;
        margin-bottom: 10px;
    }

    .card {
        padding: 20px;
        background-color: #f6f7ff;
        border-radius: 15px;
        border: 2px solid #d8dafe;
        margin-bottom: 15px;
    }

    .btn-primary button {
        background-color: #6a11cb !important;
        color: white !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============  TITLE  ============
st.markdown("<div class='title'>🌍 AI-POWERED SMART TOURISM SYSTEM</div>", unsafe_allow_html=True)

# ============  SIDEBAR NAVIGATION  ============
menu = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Home",
        "📊 Generate Dataset",
        "🤖 Train Models",
        "📈 Demand Forecast",
        "🎯 Content-Based Recommendation",
        "👥 Collaborative Filtering",
        "💬 Sentiment Prediction",
        "📁 Dataset Summary",
        "ℹ About"
    ]
)

dg = DataGenerator()
ml = MLModels()

# ============================================================
#  HOME PAGE
# ============================================================
if menu == "🏠 Home":
    st.markdown("### ✨ Welcome to the AI-powered Tourism Intelligence Dashboard")
    st.image("https://images.unsplash.com/photo-1507525428034-b723cf961d3e", use_column_width=True)
    
    st.markdown("""
    #### 🚀 Features Provided:
    - 🧭 Personalized Tourism Recommendations  
    - 📈 Demand Forecasting with Weather Analysis  
    - 🧠 Sentiment Analysis  
    - 🗂 Rich Tourism Dataset  
    - 🏨 Hotels, Events, Transportation Data  
    """)

# ============================================================
#  DATA GENERATION
# ============================================================
elif menu == "📊 Generate Dataset":
    st.markdown("<div class='section-head'>📊 Generate Full Synthetic Dataset</div>", unsafe_allow_html=True)

    if st.button("Generate Dataset", use_container_width=True):
        dg.generate_all_data()
        st.success("✅ Dataset Successfully Generated!")

# ============================================================
#  TRAIN MODELS
# ============================================================
elif menu == "🤖 Train Models":
    st.markdown("<div class='section-head'>🤖 Train All ML Models</div>", unsafe_allow_html=True)

    if st.button("Train ALL Models", use_container_width=True):
        ml.train_all_models()
        st.success("🎉 All Models Trained Successfully!")

    if st.button("Quick Train (Forecast + Recommenders + Sentiment)", use_container_width=True):
        ml.train_demand_forecasting()
        ml.train_recommendation_models()
        ml.train_sentiment_analysis()
        st.success("⚡ Quick Training Completed!")

# ============================================================
#  DEMAND FORECAST
# ============================================================
elif menu == "📈 Demand Forecast":
    st.markdown("<div class='section-head'>📈 Predict Visitor Demand</div>", unsafe_allow_html=True)

    if os.path.exists(DEST_CSV):
        dest_list = pd.read_csv(DEST_CSV)["destination"].tolist()
    else:
        dest_list = []

    col1, col2, col3 = st.columns(3)
    with col1:
        dest = st.selectbox("Select Destination", dest_list)
    with col2:
        year = st.number_input("Year", min_value=2021, max_value=2030, value=2024)
    with col3:
        month = st.slider("Month", 1, 12)

    if st.button("Predict Demand", use_container_width=True):
        output = predict_demand_for(dest, year, month)
        st.metric("Predicted Visitors", f"{output:,}")

# ============================================================
#  CONTENT BASED RECOMMENDATION
# ============================================================
elif menu == "🎯 Content-Based Recommendation":
    st.markdown("<div class='section-head'>🎯 Content-Based Destinations</div>", unsafe_allow_html=True)

    dest_list = pd.read_csv(DEST_CSV)["destination"].tolist()
    dest = st.selectbox("Choose a destination", dest_list)

    if st.button("Recommend Similar Places", use_container_width=True):
        results = content_recommendations(dest)
        st.subheader("🔍 Similar Destinations")
        for r in results:
            st.success(f"🌍 **{r['destination']}** — Similarity: `{r['score']}`")

# ============================================================
#  COLLABORATIVE FILTERING
# ============================================================
elif menu == "👥 Collaborative Filtering":
    st.markdown("<div class='section-head'>👥 User-Based Personalized Suggestions</div>", unsafe_allow_html=True)

    users = pd.read_csv(RATINGS_CSV)["user"].unique().tolist()
    user = st.selectbox("Select User", users)

    if st.button("Recommend for User", use_container_width=True):
        output = cf_recommend_for(user)
        st.subheader("🎯 Personalized Recommendations")
        for item in output:
            st.info(f"🏖 {item['destination']} — score {item['score']}")

# ============================================================
#  SENTIMENT ANALYSIS
# ============================================================
elif menu == "💬 Sentiment Prediction":
    st.markdown("<div class='section-head'>💬 Predict Review Sentiment</div>", unsafe_allow_html=True)

    text = st.text_area("Enter Review Text:")

    if st.button("Analyze Sentiment", use_container_width=True):
        result = predict_review_sentiment(text)
        st.success(f"🧠 Sentiment: **{result.upper()}**")

# ============================================================
#  DATASET SUMMARY
# ============================================================
elif menu == "📁 Dataset Summary":
    st.markdown("<div class='section-head'>📁 Dataset Summary</div>", unsafe_allow_html=True)

    files = {
        "Tourism History": HISTORY_CSV,
        "Destinations": DEST_CSV,
        "Ratings": RATINGS_CSV,
        "Reviews": REVIEWS_CSV,
        "Weather": WEATHER_CSV
    }

    for name, path in files.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            st.markdown(f"### 📌 {name}")
            st.info(f"{df.shape[0]} rows × {df.shape[1]} columns")
            st.dataframe(df.head())
        else:
            st.error(f"❌ {name} file missing!")

# ============================================================
#  ABOUT
# ============================================================
elif menu == "ℹ About":
    st.markdown("""
    ### 📝 About This Project  
    **AI-Powered Smart Tourism Recommender & Demand Optimizer**  
    **Author:** Archit Baloni  
    **Supervisor:** Dr. Lalita Chaudhary  
    **Institution:** IILM University, Greater Noida  

    Built using Machine Learning, NLP, Clustering, Forecasting, and Recommendation Systems.
    """)

