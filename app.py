import streamlit as st
import pandas as pd
import plotly.express as px

st.title("🛡️ **Aadhaar Anomaly Atlas**")
st.markdown("***UIDAI Hackathon 2026 - Fraud & Anomaly Detection***")

@st.cache_data
def load_data():
    df = pd.read_parquet('data/processed/master_anomaly_table.parquet')
    df['xgb_anomaly_prob'] = pd.read_csv('top_anomalies.csv')['xgb_anomaly_prob'].mean()  # Mock
    return df

df = load_data()

col1, col2 = st.columns(2)
with col1:
    st.metric("Total Locations", len(df))
    st.metric("Anomalies Detected", df['is_anomaly'].sum())
with col2:
    st.metric("Top Spike", f"{df['volume_spike'].max():.0f}x")
    st.metric("Fraud Risk", f"{df['is_anomaly'].mean():.1%}")

st.subheader("🔥 **Top 20 Suspicious Locations**")
top20 = df.nlargest(20, 'volume_spike')[['state', 'district', 'pincode', 'volume_spike', 'is_anomaly']]
st.dataframe(top20.style.format({'volume_spike': '{:.1f}'}))

st.subheader("📊 **Anomaly Distribution**")
fig = px.histogram(df, x='volume_spike', color='is_anomaly', nbins=50, title="Volume Spike Distribution")
st.plotly_chart(fig)

st.success("🚨 **Ready for UIDAI deployment!**")
