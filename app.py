import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Streamlit App Configuration
# ----------------------------
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

# =============================
# 🔍 Live URL Analysis Section
# =============================

st.markdown("---")
st.header("🌐 Analyze a New Webpage")

url_input = st.text_input("Enter a webpage URL to analyze:")
analyze_btn = st.button("🚀 Run Analysis")

if analyze_btn and url_input:
    try:
        st.info(f"Fetching content from: {url_input}")

        # Use headers to avoid blocking by some websites
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url_input, headers=headers, timeout=10)
        response.raise_for_status()

        # Parse HTML and remove unnecessary tags
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        # Extract visible text
        text = soup.get_text(separator=" ", strip=True)

        if not text.strip():
            st.warning("⚠️ No readable content found on this page.")
        else:
            # --- SEO Quality Metrics ---
            flesch = textstat.flesch_reading_ease(text)
            word_count = len(text.split())

            st.subheader("📊 SEO Quality Metrics")
            col1, col2 = st.columns(2)
            col1.metric("Word Count", word_count)
            col2.metric("Readability (Flesch Score)", round(flesch, 2))

            # --- Duplicate Content Check ---
            try:
                df_existing = pd.read_csv("seo-content-detector/data/extracted_content.csv")
                existing_texts = df_existing["body_text"].fillna("").tolist()

                corpus = [text] + existing_texts
                vectorizer = TfidfVectorizer().fit_transform(corpus)
                similarity_matrix = cosine_similarity(vectorizer[0:1], vectorizer[1:])
                max_sim = similarity_matrix.max()

                st.subheader("🔁 Duplicate Content Check")
                st.write(f"**Most Similar Existing Page:** {max_sim * 100:.2f}% match")

                if max_sim > 0.8:
                    st.error("⚠️ High duplicate risk detected! The content is very similar to existing pages.")
                else:
                    st.success("✅ This content appears unique!")

            except Exception:
                st.warning("⚠️ Could not compare with existing dataset. Please ensure 'data/extracted_content.csv' exists.")

    except Exception as e:
        st.error(f"❌ Error analyzing URL: {e}")

else:
    st.info("Enter a URL above and click **Run Analysis** to begin.")
