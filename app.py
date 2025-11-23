import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------- Page Config -----------------
st.set_page_config(page_title="Road Traffic Predictor", 
                   layout="wide",
                   initial_sidebar_state="expanded")

# ----------------- Load Trained Model -----------------
model = load("traffic_model_app_pro.joblib")

# ----------------- Sidebar -----------------
st.sidebar.header("Traffic Prediction Inputs")
st.sidebar.markdown("Adjust the parameters below to predict traffic volume:")

hour = st.sidebar.slider("Hour of the day", 0, 23, 8)
dayofweek = st.sidebar.selectbox("Day of the Week", 
                                 ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
dayofweek_num = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(dayofweek)
month = st.sidebar.selectbox("Month", list(range(1,13)))
day_of_year = st.sidebar.slider("Day of Year", 1, 366, 1)
is_weekend = 1 if dayofweek_num >= 5 else 0
temp = st.sidebar.number_input("Temperature (°C)", -30.0, 50.0, 25.0)
rain_1h = st.sidebar.number_input("Rainfall in last 1 hour (mm)", 0.0, 500.0, 0.0)
clouds_all = st.sidebar.number_input("Cloud coverage (%)", 0, 100, 20)
is_holiday = st.sidebar.selectbox("Holiday?", ["No","Yes"])
is_holiday = 1 if is_holiday=="Yes" else 0
distance = st.sidebar.number_input("Distance from city center (km)", 0.0, 100.0, 10.0)

# ----------------- Prepare Input Data -----------------
input_df = pd.DataFrame({
    "hour":[hour],
    "dayofweek":[dayofweek_num],
    "month":[month],
    "day_of_year":[day_of_year],
    "is_weekend":[is_weekend],
    "temp":[temp],
    "rain_1h":[rain_1h],
    "clouds_all":[clouds_all],
    "is_holiday":[is_holiday],
    "distance":[distance]
})

# ----------------- Tabs Layout -----------------
tab1, tab2 = st.tabs(["Prediction", "Visualization"])

# ----------------- Tab 1: Prediction -----------------
with tab1:
    st.subheader("Predict Traffic Volume")
    if st.button("Predict"):
        pred = model.predict(input_df)
        st.success(f"🚦 Predicted Traffic Volume: **{int(pred[0])}** vehicles/hour")
    
    st.markdown("""
    **Tips for Better Predictions:**
    - Choose accurate hour, weather, and holiday info.  
    - Distance can be set relative to the city center or main traffic hubs.  
    - Use the visualization tab to explore trends in traffic data.
    """)

# ----------------- Tab 2: Visualization -----------------
with tab2:
    st.subheader("Explore Traffic Trends")
    uploaded_file = st.file_uploader("Upload your CSV (e.g., Metro_Interstate_Traffic_Volume.csv)", type="csv")
    
    if uploaded_file:
        df_csv = pd.read_csv(uploaded_file)
        
        st.markdown("### Data Preview")
        st.dataframe(df_csv.head())
        
        # Prepare datetime columns if exist
        if 'date_time' in df_csv.columns:
            df_csv['date_time'] = pd.to_datetime(df_csv['date_time'])
            df_csv['hour'] = df_csv['date_time'].dt.hour
            df_csv['dayofweek'] = df_csv['date_time'].dt.dayofweek
            df_csv['month'] = df_csv['date_time'].dt.month
        
        sns.set(style="whitegrid", palette="muted")
        
        # Avg Traffic by Hour
        st.markdown("### Average Traffic Volume by Hour")
        fig, ax = plt.subplots(figsize=(10,4))
        sns.barplot(x='hour', y='traffic_volume', data=df_csv.groupby('hour')['traffic_volume'].mean().reset_index(), ax=ax)
        st.pyplot(fig)
        
        # Avg Traffic by Day of Week
        st.markdown("### Average Traffic Volume by Day of Week")
        fig, ax = plt.subplots(figsize=(10,4))
        sns.barplot(x='dayofweek', y='traffic_volume', data=df_csv.groupby('dayofweek')['traffic_volume'].mean().reset_index(), ax=ax)
        st.pyplot(fig)
        
        # Traffic vs Temperature (if exists)
        if 'temp' in df_csv.columns:
            st.markdown("### Traffic Volume vs Temperature")
            fig, ax = plt.subplots(figsize=(10,4))
            sns.scatterplot(x='temp', y='traffic_volume', data=df_csv, hue='dayofweek', palette='tab10', ax=ax)
            st.pyplot(fig)
        
        # Correlation Heatmap
        st.markdown("### Correlation Heatmap (numeric features)")
        numeric_cols = df_csv.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(10,6))
            sns.heatmap(df_csv[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)
    else:
        st.info("📌 Upload a CSV file to see visualizations here.")
