# SEO Content Quality & Duplicate Detector

## 1. Project Overview
This project is a complete machine learning pipeline, contained within a Jupyter Notebook, that analyzes web content for SEO quality and duplication. It scrapes content from a list of URLs, extracts NLP features, detects near-duplicate articles, and uses a classification model to score content quality as Low, Medium, or High.

## 2. Setup and Run
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/nagashreem363/SEO-Content-Quality-Duplicate-Detector.git
    cd seo-content-detector
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the pipeline:**
    *   Open the Jupyter Notebook: `notebooks/seo_pipeline.ipynb`
    *   Run all cells from top to bottom. The script will automatically download the necessary dataset and create the `data/` and `models/` folders with the output.

## 3. Key Decisions & Results
*   **HTML Parsing & Embeddings:** Used `BeautifulSoup4` for robust parsing and `TF-IDF vectors` for efficient similarity detection.
*   **Quality Model:** A `Random Forest Classifier` was trained on features like word count and readability. It achieved **84.0% accuracy**, outperforming a 72.0% baseline.
*   **Key Findings:** The analysis of 81 URLs identified **8 duplicate pairs** and flagged **37%** of the pages as "thin content."

## 4. Limitations
*   The model's quality labels are synthetic (rule-based) and may not perfectly reflect human judgment.
*   The analysis is limited to on-page text and does not consider other SEO factors like backlinks or page speed.
##  5.Results Summary

*Dataset: 81 URLs analyzed

*Duplicate Pairs: 8

*Thin Content Pages: 28 (34.6%)

Metric	Precision	Recall	F1-score	Support
Low	0.923	1.000	0.960	12
Medium	0.875	0.875	0.875	8
High	1.000	0.800	0.889	5
Overall Accuracy	—	—	—	0.920

*Sample Duplicate URLs:

URL 1	URL 2	Similarity
https://guardiandigital.com/
...	https://inspiredelearning.com/
...	0.80
https://microsoft.com/
...	https://zscaler.com/
...	0.85
https://sign.dropbox.com/
...	https://fax.plus/
...	0.81
