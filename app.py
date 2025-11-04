import streamlit as st
import pandas as pd

st.set_page_config(page_title="SEO Content Quality & Duplicate Detector", layout="wide")

st.title("🧠 SEO Content Quality & Duplicate Detector")
st.write("Analyze your website content for quality, duplication, and readability.")

st.divider()

st.subheader("📂 Available Data Outputs")

tabs = st.tabs(["Extracted Content", "Features", "Duplicates"])

# --- Tab 1: Extracted content ---
with tabs[0]:
    try:
        df1 = pd.read_csv("seo-content-detector/data/extracted_content.csv")
        st.dataframe(df1)
    except Exception as e:
        st.warning("Extracted content file not found. Please run the pipeline first.")

# --- Tab 2: Features ---
with tabs[1]:
    try:
        df2 = pd.read_csv("seo-content-detector/data/features.csv")
        st.dataframe(df2)
    except Exception as e:
        st.warning("Features file not found. Please run the pipeline first.")

# --- Tab 3: Duplicates ---
with tabs[2]:
    try:
        df3 = pd.read_csv("seo-content-detector/data/duplicates.csv")
        st.dataframe(df3)
    except Exception as e:
        st.warning("Duplicates file not found. Please run the pipeline first.")

st.success("App loaded successfully! 🚀")
