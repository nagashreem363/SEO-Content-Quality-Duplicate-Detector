#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[3]:


# =========================
# STEP 0: INSTALL DEPENDENCIES
# =========================
try:
    get_ipython  
    IN_NOTEBOOK = True
except Exception:
    IN_NOTEBOOK = False


# In[4]:


#def pip_install(pkg_line):
 #   import sys, subprocess
  #  subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkg_line.split())


# In[5]:


for pkgs in [
    "beautifulsoup4==4.12.3",
    "requests==2.32.3",
    "pandas==2.2.2",
    "numpy==1.26.4",
    "scikit-learn==1.4.2",
    "textstat==0.7.4",
    "tqdm==4.66.4",
    "lxml==5.2.2",
    "gdown==5.1.0",
]:
    pip_install(pkgs)


# In[6]:


import os
import re
import json
import time
import random
import pickle
import warnings
warnings.filterwarnings("ignore")


# In[7]:


import numpy as np
import pandas as pd
import requests


# In[8]:


from tqdm import tqdm
from bs4 import BeautifulSoup


# In[9]:


import textstat


# In[10]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# In[11]:


from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# In[ ]:





# In[12]:


BASE_DIR = "seo-content-detector"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")


# In[13]:


os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)


# In[14]:


CONFIG = {
    "duplicate_threshold": 0.80,
    "similarity_threshold_realtime": 0.75,
    "thin_content_threshold": 500,
    "keywords_top_n": 5,
    "tfidf_max_features_keywords": 5000,
    "tfidf_max_features_embeddings": 2000,
    "scrape_user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "scrape_timeout": 20,
    "scrape_delay_min_sec": 1.0,
    "scrape_delay_max_sec": 2.0,
    "train_test_split_ratio": 0.30,
    "random_state": 42,
    "alternative_dataset_gdrive_id": "1q-49ey_ydbB1TnN5x60K9VhyoM0vDI1K",
}
random.seed(CONFIG["random_state"])
np.random.seed(CONFIG["random_state"])


# In[15]:


with open(os.path.join(BASE_DIR, "config.json"), "w") as f:
    json.dump(CONFIG, f, indent=2)


# In[16]:


print("Config saved to:", os.path.join(BASE_DIR, "config.json"))


# In[17]:


# =========================
# STEP 2: DATA LOADING / DOWNLOADING
# =========================
def ensure_dataset():
    """
    Use local seo-content-detector/data/data.csv if present.
    Otherwise auto-download the alternative URLs-only dataset via Google Drive (gdown).
    Expected columns:
      - Primary: url, html_content
      - Alternative: url
    """
    target_csv = os.path.join(DATA_DIR, "data.csv")
    if os.path.exists(target_csv):
        df = pd.read_csv(target_csv)
        print(f"Found dataset at {target_csv} (rows={len(df)})")
        return df

    print("data.csv not found. Attempting to download alternative URLs-only dataset via Google Drive...")
    import gdown
    g_url = f"https://drive.google.com/uc?id={CONFIG['alternative_dataset_gdrive_id']}"
    alt_path = os.path.join(DATA_DIR, "urls_only.csv")
    gdown.download(g_url, alt_path, quiet=False)
    dfa = pd.read_csv(alt_path)

    # Normalize column to 'url'
    if "url" not in dfa.columns:
        for cand in ["URL", "Url", "link", "Link", "urls", "Urls", "Links"]:
            if cand in dfa.columns:
                dfa = dfa.rename(columns={cand: "url"})
                break
    if "url" not in dfa.columns:
        raise RuntimeError("Downloaded alternative dataset doesn't contain a 'url' column.")

    dfa.to_csv(target_csv, index=False)
    print(f"Alternative dataset normalized and saved to {target_csv} (rows={len(dfa)})")
    return dfa


# In[18]:


def infer_html_column(df):
    cols = list(df.columns)
    lc = [c.lower() for c in cols]
    for needle in ["html_content", "html", "raw_html", "content_html"]:
        if needle in lc:
            return cols[lc.index(needle)]
    for i, name in enumerate(lc):
        if "html" in name:
            return cols[i]
    for i, name in enumerate(lc):
        if "content" in name:
            return cols[i]
    return None


# In[19]:


# =========================
# STEP 3: SESSION, SCRAPING & HTML PARSING
# =========================
def make_session():
    sess = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess


# In[20]:


SESSION = make_session()


# In[21]:


def scrape_url(url, headers=None, timeout=15):
    headers = headers or {"User-Agent": CONFIG["scrape_user_agent"], "Accept-Language": "en-US,en;q=0.9"}
    try:
        resp = SESSION.get(url, headers=headers, timeout=timeout)
        if resp.status_code == 200 and resp.text:
            return resp.text
        return None
    except requests.RequestException:
        return None


# In[22]:


def extract_main_text(html):
    """
    Extract title and main content with heuristics.
    """
    if not html or not isinstance(html, str):
        return "", ""

    try:
        soup = BeautifulSoup(html, "lxml")

        # Title
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # Remove boilerplate
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
            tag.decompose()

        # Preferred containers
        candidates = []
        selectors = ["article", "main", '[role="main"]', "div#content", "div.content", "div.post", "section"]
        for sel in selectors:
            for node in soup.select(sel):
                txt = node.get_text(separator=" ", strip=True)
                wc = len(txt.split())
                candidates.append((txt, wc))

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            body_text = candidates[0][0]
        else:
            # Fallback: p/h*/li
            texts = []
            for tag in soup.find_all(["p", "h1", "h2", "h3", "li"]):
                t = tag.get_text(" ", strip=True)
                if t:
                    texts.append(t)
            body_text = " ".join(texts)

        # Final fallback if still short
        if len(body_text.split()) < 30:
            full_text = soup.get_text(separator=" ", strip=True)
            if len(full_text.split()) > len(body_text.split()):
                body_text = full_text

        body_text = re.sub(r"\s+", " ", body_text).strip()
        return title, body_text
    except Exception:
        return "", ""


# In[23]:


def parse_primary_dataset(df, html_col):
    rows = []
    empty_count = 0
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Parsing HTML (primary)"):
        url = r.get("url", "")
        html = r.get(html_col, None)
        title, body_text = extract_main_text(html)
        wc = len(body_text.split()) if body_text else 0
        if wc == 0:
            empty_count += 1
        rows.append({"url": url, "title": title, "body_text": body_text, "word_count": wc})

    out = pd.DataFrame(rows)
    out_path = os.path.join(DATA_DIR, "extracted_content.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved extracted content to {out_path} (rows={len(out)})")
    print(f"Empty/zero-word pages (primary): {empty_count}/{len(out)}")
    return out


# In[24]:


def parse_alternative_dataset_and_scrape(df_urls):
    assert "url" in df_urls.columns, "Alternative dataset must contain 'url' column."
    rows = []
    ok = 0
    for _, r in tqdm(df_urls.iterrows(), total=len(df_urls), desc="Scraping & parsing (alternative)"):
        url = str(r["url"])
        html = scrape_url(url, headers={"User-Agent": CONFIG["scrape_user_agent"], "Accept-Language": "en-US,en;q=0.9"}, timeout=CONFIG["scrape_timeout"])
        if html:
            title, body_text = extract_main_text(html)
        else:
            title, body_text = "", ""
        wc = len(body_text.split()) if body_text else 0
        ok += (wc > 0)
        rows.append({"url": url, "title": title, "body_text": body_text, "word_count": wc})
        time.sleep(random.uniform(CONFIG["scrape_delay_min_sec"], CONFIG["scrape_delay_max_sec"]))

    out = pd.DataFrame(rows)
    out_path = os.path.join(DATA_DIR, "extracted_content.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved extracted content to {out_path} (rows={len(out)}, non-empty={ok})")
    return out


# In[25]:


# =========================
# STEP 4: TEXT CLEANING & FEATURES
# =========================
def clean_text_simple(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s.strip()


# In[26]:


def count_sentences(text: str) -> int:
    if not text or not isinstance(text, str):
        return 0
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    parts = [p for p in parts if p.strip()]
    return len(parts)


# In[27]:


def compute_readability(s: str) -> float:
    try:
        return float(textstat.flesch_reading_ease(s or ""))
    except Exception:
        return 0.0


# In[28]:


def extract_keywords_tfidf(texts, top_n=5, max_features=5000):
    if all((not t) or t.strip() == "" for t in texts):
        return [""] * len(texts), None
    vec = TfidfVectorizer(max_features=max_features, stop_words="english")
    tfidf = vec.fit_transform(texts)
    feature_names = np.array(vec.get_feature_names_out())
    result = []
    for i in range(tfidf.shape[0]):
        row = tfidf.getrow(i).toarray().ravel()
        if row.sum() == 0:
            result.append("")
            continue
        top_idx = row.argsort()[-top_n:][::-1]
        kws = feature_names[top_idx]
        result.append("|".join(kws))
    with open(os.path.join(MODELS_DIR, "keywords_tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vec, f)
    return result, vec


# In[29]:


def compute_embeddings_tfidf(texts, max_features=2000):
    if all((not t) or t.strip() == "" for t in texts):
        return None, None
    vec = TfidfVectorizer(max_features=max_features, stop_words="english")
    X = vec.fit_transform(texts)
    emb = X.toarray().astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    emb = emb / norms
    with open(os.path.join(MODELS_DIR, "tfidf_embed_vectorizer.pkl"), "wb") as f:
        pickle.dump(vec, f)
    return emb, vec


# In[30]:


def extract_features(extracted_df):
    df = extracted_df.copy()
    df["clean_text"] = df["body_text"].fillna("").astype(str).apply(clean_text_simple)
    df["sentence_count"] = df["body_text"].fillna("").astype(str).apply(count_sentences)
    df["flesch_reading_ease"] = df["body_text"].fillna("").astype(str).apply(compute_readability)

    # Keywords
    df["top_keywords"], _ = extract_keywords_tfidf(
        df["clean_text"].tolist(),
        top_n=CONFIG["keywords_top_n"],
        max_features=CONFIG["tfidf_max_features_keywords"]
    )

    # Embeddings
    embeddings, embed_vec = compute_embeddings_tfidf(
        df["clean_text"].tolist(),
        max_features=CONFIG["tfidf_max_features_embeddings"]
    )
    if embeddings is None:
        print("All texts are empty. Cannot compute embeddings. Aborting further steps.")
        df["embedding"] = "[]"
        out_path = os.path.join(DATA_DIR, "features.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved features to {out_path} (rows={len(df)})")
        return df, None

    df["embedding"] = [json.dumps(v.tolist()) for v in embeddings]
    out_path = os.path.join(DATA_DIR, "features.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved features to {out_path} (rows={len(df)})")
    return df, embeddings


# In[31]:


# =========================
# STEP 5: DUPLICATE DETECTION
# =========================
def detect_duplicates(features_df, embeddings, threshold=0.80):
    if embeddings is None or len(embeddings) == 0:
        print("No embeddings available. Skipping duplicate detection.")
        dup_df = pd.DataFrame(columns=["url1","url2","similarity"])
        dup_df.to_csv(os.path.join(DATA_DIR, "duplicates.csv"), index=False)
        return dup_df

    sim = cosine_similarity(embeddings)
    n = sim.shape[0]

    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            s = float(sim[i, j])
            if s > threshold:
                pairs.append({
                    "url1": features_df.iloc[i]["url"],
                    "url2": features_df.iloc[j]["url"],
                    "similarity": round(s, 4)
                })

    dup_df = pd.DataFrame(pairs)
    out_path = os.path.join(DATA_DIR, "duplicates.csv")
    dup_df.to_csv(out_path, index=False)
    print(f"Saved duplicate pairs to {out_path} (pairs={len(dup_df)})")

    features_df["is_thin"] = features_df["word_count"] < CONFIG["thin_content_threshold"]
    features_df.to_csv(os.path.join(DATA_DIR, "features.csv"), index=False)

    total = len(features_df)
    thin_count = int(features_df["is_thin"].sum())
    print("Summary:")
    print(f"- Total pages analyzed: {total}")
    print(f"- Duplicate pairs: {len(dup_df)}")
    print(f"- Thin content pages: {thin_count} ({100.0 * thin_count / max(1,total):.1f}%)")

    if len(dup_df) > 0:
        print("Sample duplicates:")
        print(dup_df.head(5).to_string(index=False))

    return dup_df


# In[32]:


# =========================
# STEP 6: QUALITY MODEL
# =========================
def label_quality_rules(row):
    wc = row["word_count"]
    rd = row["flesch_reading_ease"]
    if (wc > 1500) and (50 <= rd <= 70):
        return "High"
    if (wc < 500) or (rd < 30):
        return "Low"
    return "Medium"


# In[33]:


def make_baseline_prediction(word_count):
    if word_count > 1500:
        return "High"
    if word_count < 500:
        return "Low"
    return "Medium"


# In[34]:


def ensure_label_variety(df):
    labels, counts = np.unique(df["quality_label"].values, return_counts=True)
    if len(labels) >= 2:
        return df, "rules"
    wc = df["word_count"].values
    rd = df["flesch_reading_ease"].values
    q80_wc = np.percentile(wc, 80)
    q20_wc = np.percentile(wc, 20)
    q60_rd = np.percentile(rd, 60)
    new_labels = []
    for i in range(len(df)):
        if (wc[i] >= q80_wc) and (rd[i] >= q60_rd):
            new_labels.append("High")
        elif (wc[i] <= q20_wc) or (rd[i] < 30):
            new_labels.append("Low")
        else:
            new_labels.append("Medium")
    df["quality_label"] = new_labels
    return df, "quantile-fallback"


# In[35]:


def train_quality_model(features_df):
    df = features_df.copy()
    df["quality_label"] = df.apply(label_quality_rules, axis=1)
    df, label_strategy = ensure_label_variety(df)

    feat_cols = ["word_count", "sentence_count", "flesch_reading_ease"]
    X = df[feat_cols].values
    y = df["quality_label"].values

    labels, counts = np.unique(y, return_counts=True)
    can_stratify = all(c >= 2 for c in counts) and len(labels) >= 2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG["train_test_split_ratio"],
        random_state=CONFIG["random_state"],
        stratify=y if can_stratify else None
    )

    if len(np.unique(y_train)) < 2:
        X_train = np.vstack([X_train, X_test])
        y_train = np.concatenate([y_train, y_test])

    if len(np.unique(y_train)) < 2:
        print("Not enough label variety to train classifier. Skipping ML model training.")
        return None, 0.0, 0.0, []

    model = RandomForestClassifier(n_estimators=200, random_state=CONFIG["random_state"])
    model.fit(X_train, y_train)

    with open(os.path.join(MODELS_DIR, "quality_model.pkl"), "wb") as f:
        pickle.dump(model, f)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    labels_order = ["Low", "Medium", "High"]
    rep = classification_report(y_test, y_pred, labels=labels_order, zero_division=0, digits=3)
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)

    baseline_pred = [make_baseline_prediction(wc) for wc in X_test[:, 0]]
    baseline_acc = accuracy_score(y_test, baseline_pred)

    importances = list(zip(feat_cols, model.feature_importances_))
    importances.sort(key=lambda x: x[1], reverse=True)

    print(f"Labeling strategy: {label_strategy}")
    print("Model Performance:")
    print(rep)
    print(f"Overall Accuracy: {acc:.3f}")
    print(f"Baseline Accuracy (word_count only): {baseline_acc:.3f}")
    print("Confusion Matrix [Low, Medium, High]:")
    print(cm)
    print("Top Features:")
    for name, imp in importances[:3]:
        print(f"- {name}: {imp:.3f}")

    return model, acc, baseline_acc, importances


# In[36]:


# =========================
# STEP 7: REAL-TIME ANALYSIS
# =========================
def analyze_url(url):
    """
    Scrapes URL, extracts features, predicts quality, and finds similar content.
    """
    headers = {"User-Agent": CONFIG["scrape_user_agent"], "Accept-Language": "en-US,en;q=0.9"}
    try:
        resp = SESSION.get(url, headers=headers, timeout=CONFIG["scrape_timeout"])
        resp.raise_for_status()
        title, body_text = extract_main_text(resp.text)
    except Exception as e:
        return {"error": f"Failed to fetch URL: {e}"}

    if not body_text:
        return {"error": "No content extracted from the URL."}

    clean = clean_text_simple(body_text)
    word_count = len(body_text.split())
    sentence_count = count_sentences(body_text)
    readability = compute_readability(body_text)
    is_thin = word_count < CONFIG["thin_content_threshold"]

    model_path = os.path.join(MODELS_DIR, "quality_model.pkl")
    if not os.path.exists(model_path):
        return {"error": "quality_model.pkl not found. Run training first."}
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    features = np.array([[word_count, sentence_count, readability]], dtype=np.float32)
    quality_label = model.predict(features)[0]

    vec_path = os.path.join(MODELS_DIR, "tfidf_embed_vectorizer.pkl")
    if not os.path.exists(vec_path):
        return {"error": "tfidf_embed_vectorizer.pkl not found. Run feature extraction first."}
    with open(vec_path, "rb") as f:
        embed_vec = pickle.load(f)

    emb_query = embed_vec.transform([clean]).toarray().astype(np.float32)
    emb_query = emb_query / (np.linalg.norm(emb_query, axis=1, keepdims=True) + 1e-8)

    feat_path = os.path.join(DATA_DIR, "features.csv")
    if not os.path.exists(feat_path):
        return {"error": "features.csv not found. Run the pipeline first."}
    existing = pd.read_csv(feat_path)

    try:
        existing_embeddings = np.vstack([np.array(json.loads(x), dtype=np.float32) for x in existing["embedding"].tolist()])
    except Exception:
        existing_embeddings = []
        for e_str in existing["embedding"].tolist():
            try:
                vec = json.loads(e_str)
            except Exception:
                vec = eval(e_str)
            existing_embeddings.append(np.array(vec, dtype=np.float32))
        existing_embeddings = np.vstack(existing_embeddings)

    sims = cosine_similarity(emb_query, existing_embeddings)[0]
    similar = []
    for i, s in enumerate(sims):
        if s > CONFIG["similarity_threshold_realtime"]:
            similar.append({"url": existing.iloc[i]["url"], "similarity": float(s)})
    similar = sorted(similar, key=lambda d: d["similarity"], reverse=True)[:5]

    return {
        "url": url,
        "title": title,
        "word_count": int(word_count),
        "sentence_count": int(sentence_count),
        "readability": float(readability),
        "quality_label": str(quality_label),
        "is_thin": bool(is_thin),
        "similar_to": similar,
    }


# In[37]:


# =========================
# STEP 8: RUN PIPELINE END-TO-END
# =========================
def run_pipeline():
    print("=== SEO Content Quality & Duplicate Detector ===")
    # 1) Load dataset (local or auto-download alternative)
    df_raw = ensure_dataset()
    if len(df_raw) == 0:
        raise RuntimeError("Dataset is empty.")

    df_raw.to_csv(os.path.join(DATA_DIR, "data.csv"), index=False)

    # 2) Parse: primary or alternative
    if "url" not in df_raw.columns:
        raise RuntimeError("Dataset must contain 'url' column.")
    html_col = infer_html_column(df_raw)
    if html_col and df_raw[html_col].notna().sum() > 0:
        print(f"Using primary dataset (HTML column='{html_col}')")
        df_content = parse_primary_dataset(df_raw, html_col)
    else:
        print("Using alternative dataset (URLs only). Scraping pages...")
        df_content = parse_alternative_dataset_and_scrape(df_raw[["url"]])

    if df_content["word_count"].sum() == 0:
        print("All extracted pages are empty. Check accessibility or provide primary dataset with html_content.")
        return

    # 3) Features
    features_df, embeddings = extract_features(df_content)
    if embeddings is None:
        print("No embeddings could be computed (empty text). Stopping after features.")
        return

    # 4) Duplicates + thin content
    dup_df = detect_duplicates(features_df, embeddings, threshold=CONFIG["duplicate_threshold"])

    # 5) Model training + evaluation
    model, acc, baseline_acc, importances = train_quality_model(features_df)

    # 6) Real-time demo (first URL)
    test_url = features_df.iloc[0]["url"]
    print("\nReal-time analyze_url() demo on first dataset URL:")
    result = analyze_url(test_url)
    print(json.dumps(result, indent=2))

    print("\nPipeline complete. Outputs saved in seo-content-detector/data and models/")
    print("- data/extracted_content.csv")
    print("- data/features.csv")
    print("- data/duplicates.csv")
    print("- models/quality_model.pkl")
    print("- models/tfidf_embed_vectorizer.pkl")


# In[39]:


# Run everything
run_pipeline()

