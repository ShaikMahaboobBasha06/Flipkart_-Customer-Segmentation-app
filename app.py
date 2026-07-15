import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
# 1. Page Configuration
st.set_page_config(page_title="Flipkart Customer Segmentation Portal", layout="wide")
st.title("🛍️ Flipkart Customer Segmentation Portal (POC)")
st.write("An interactive web deployment translating your K-Means notebook into a live operational app.")
# 2. Sidebar Data Input
st.sidebar.header("📁 Data Source Configuration")
uploaded_file = st.sidebar.file_uploader("Upload custom customer CSV data", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Custom dataset integrated!")
else:
    st.sidebar.info("Populating environment with your 5-row infographic sample data.")
    # Target features from your POC
    data = {
        'Customer_ID': [101, 102, 103, 104, 105],
        'Annual_Spending': [5000, 15000, 20000, 3000, 25000],
        'Orders_Count': [2, 5, 7, 1, 8]
    }
    df = pd.DataFrame(data)
# Toggle View Layer Data Summary Matrix
if st.checkbox("🔍 Inspect Active Customer Matrix Data Profiles"):
    st.write("### Data Distribution Model Summary")
    st.dataframe(df.describe())
    st.write("### Raw Data View")
    st.dataframe(df)
# 3. Model Variable Selection & Pipeline Parameters
st.sidebar.header("🤖 K-Means Configuration")
num_clusters = st.sidebar.slider("Select Dynamic Clusters (K)", min_value=2, max_value=5, value=3)
# Features targeted by your notebook workflow
features = ['Annual_Spending', 'Orders_Count']
X = df[features]
# Preprocessing: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Model Fitting
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df['Segment_ID'] = kmeans.fit_predict(X_scaled)
df['Segment_ID'] = df['Segment_ID'].map(lambda x: f"Segment {x + 1}")  # Readable Labels
# 4. Display Layer: Upgraded Dynamic Graph Layouts
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("📊 Dynamic Segment Clusters Graph View")
    fig = px.scatter(
        df,
        x='Annual_Spending',
        y='Orders_Count',
        color='Segment_ID',
        hover_data=['Customer_ID'],
        title="Interactive Customer Cohort Projections",
        labels={'Annual_Spending': 'Annual Revenue Generation (₹)', 'Orders_Count': 'Completed Orders Volume'},
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig.update_traces(marker=dict(size=14, line=dict(width=1.5, color='White')))
    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.subheader("📋 Segment Allocation Matrix")
    st.dataframe(df.sort_values(by='Segment_ID')[['Customer_ID', 'Segment_ID', 'Annual_Spending', 'Orders_Count']])
    st.subheader("📈 Profiling Averages")
    summary = df.groupby('Segment_ID')[features].mean()
    st.table(summary)