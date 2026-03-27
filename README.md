# Customer Feedback Intelligence Pipeline
## Overview

The **Customer Feedback Intelligence Pipeline** is a data science system that analyzes large volumes of customer reviews for the **Robinhood** mobile application to extract useful business insights. The pipeline processes raw customer reviews in text format, converts it into numerical features, applies ML techniques to identify recurring themes and measure customer sentiment.

Goal: Successfully identify the top 5 most frequently occurring complaint themes in Robinhood app store reviews, and classify review sentiment (positive/negative/neutral) with at least 75% accuracy, allowing product teams to prioritize which pain points to address.

## Problem Statement

Companies receive thousands of customer reviews, support tickets, and feedback comments. Manually reading and summarizing this feedback is time-consuming and often leads to missed trends, delayed responses to issues.

This project addresses that problem by building a pipeline that:

* Identifies the top 5 recurring complaint themes using unsupervised clustering
* Classifies customer sentiment (positive/negative/neutral)
* Highlights main product pain points (a cluster that contains at least 5% of reviews and has a majority of negative reviews)

---
## How to Run

### Prerequisites
- Python 3.xx+
- Google Colab or Jupyter Notebook

### Installation
``` pip install app-store-web-scraper google-play-scraper langdetect scikit-learn pandas matplotlib seaborn ```

### Running the Pipeline
Open `robinhood_reviews_cs506_project.ipynb` in Google Colab and run all cells from the top. Each section is labeled and can be run independently after the data collection cells have been ran.

The notebook will:
1. Scrape reviews from the App Store and Google Play
2. Combine, clean and save the data as `robinhood_reviews_cleaned.csv`
3. Generate all EDA visualizations
4. Train and evaluate models

### Current Data Sources

| Source | Library | Reviews Collected | Notes |
|--------|---------|-------------------|-------|
| Apple App Store (US) | `app-store-web-scraper` | ~500 | Apple's public API caps at 500 reviews per country |
| Apple App Store (GB) | `app-store-web-scraper` | ~100 | Stopped early due to API returning bad entries |
| Google Play Store (US) | `google-play-scraper` | ~5,000 |  |
| **Total (after cleaning)** | | **~4,483** | After deduplication and non-English removal |

### Data Fields Collected
- `review` - raw review text
- `rating` - star rating (1–5)
- `date` - review submission date
- `title` - review title
- `country` - country code or 'google_play'
- `sentiment` - derived label (negative/neutral/positive)
- `source` - 'App Store' or 'Google Play'

---

## Visualizations

| Visualization | Insight |
|--------------|---------|
| Rating Distribution | Heavily skewed; most reviews are 1-star or 5-star, fewer in betweens |
| Sentiment Distribution | More negative than positive reviews in general |
| Review Length Distribution | Most reviews are short (under 200 characters) |
| Avg Review Length by Sentiment | Negative reviews tend to be longer since users write more when complaining |
| Review Volume by Source | ~89% Google Play, ~11% App Store; dataset skews toward Android users |
| Average Rating Over Time | Ratings trending downward from ~3.5 in mid-2025 to ~2.0-2.5 in early 2026 |
| Review Volume Over Time | Low App Store volume causes misleading spikes in early period |

---

## Data Processing

### Cleaning Steps
1. **Remove duplicatiion** — removed duplicate reviews using exact text match
2. **Non-English removal** — used `langdetect` to filter non-English reviews; and reviews shorter than 20-30 characters were kept because of reliability
3. **Text normalization** — lowercased all text, removed numbers and punctuation using regulr expression
4. **Stop word removal** — handled using sklearn's `ENGLISH_STOP_WORDS` plus domain specific stopwords added

---

## Feature Extraction (TF-IDF)

Text reviews are converted to numerical features using **TF-IDF** whihch is relevant. It emphasizes the word that are important to specific reviews and reduced the weights on words that appear across all reviews (because they could be stop words).

**Parameters:**
- `max_features=1000` — top 1000 most useful terms
- `ngram_range=(1,2)` — captures single words and two-word phrases (e.g. "customer service")
- `min_df=3` — a term must appear in at least 3 reviews to be included
- `max_df=0.9` — terms appearing in more than 90% of reviews are excluded
- `sublinear_tf=True` — replaces raw frequency with 1 + log(freq) to reduce the impact of very common terms

**Result:** TF-IDF matrix of shape (4483, 1000) — 4483 reviews × 1000 features

---

## Dimensionality Reduction (LSA)

Before clustering, TF-IDF features are reduced using **Latent Semantic Analysis (LSA)** via `TruncatedSVD` becuase K-Means makes spherical clusters, but text data in a high-dimensional TF-IDF space is sparse and not spherical.

- `n_components=20` was chosen after experimenting

**Result:** Reduced matrix of shape (4483, 20)

---

## Challenges Faced

1. **Unbalanced data volume by source**: 89% Google Play, 11% App Store. Results dont full represent IOS users
2. **Apple API cap**- the app store scraper API limits to 500 reviews per country (10 pages × 50 reviews), regardless of how many are requested
3. **Neutral class is small**: only ~300 neutral reviews/3 star rating
4. **Silhouette scores are low**: due to high dimensionality and sparse overlap between topics. LSA improved scores significantly (from ~0.02 to ~0.18) but did not reach the 0.30 target
5. **Recent data only**: Older complaints or long-term trends are not captured covers; May 2025–March 2026.

---


## Project Goals

| # | Goal | Metrics |
|---|------|------------------------|
| 1 | Identify the top 5 recurring complaint themes (pain point) | K-Means clustering with silhouette score > 0.30 if possible |
| 2 | Classify review sentiment (positive / negative / neutral) | Logistic Regression with > 75% accuracy |
| 3 | Rank pain points by frequency | Frequency count of reviews per cluster |
| 4 | Distinguish features by rating | Compare most common terms in 1–2 star vs. 4–5 star reviews |

**If data is limited:** a Kaggle review dataset would be used instead (like [Amazon product reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)) using the same pipeline.


## Dataset

### Primary Data Source

-  **App Store reviews for Robinhood** scraped using the [app-store-web-scraper](https://pypi.org/project/app-store-web-scraper/) Python library (no API key neded)
  - Target: ~5,000–10,000 English reviews
  - Fields collected: review text, star rating, review date, review title
  - No API key or authentication required
- Amazon product review datasets (Kaggle as fallback)

### Data Collection

Primary method:

* `app-store-web-scraper` Python library (for App Store data)

```python
from app_store_web_scraper import AppStore

app = AppStoreEntry(app_id=938003185, country="us")
```

---

## Modeling Plan

### 1. Text Feature Extraction
- Tokenize and lowercase review text
- Remove stopwords, punctuation, and non-English characters
- Apply **TF-IDF vectorization** (top 500–1000 features) to convert text to numerical vectors

### 2. Clustering 
- Apply **K-Means clustering** (k = 5–10, chosen by silhouette score)
- Label each cluster manually by inspecting its top TF-IDF terms
- Metric: reported with silhouette score 

### 3. Sentiment Classification
- Labels derived from star ratings:
  - 1–2 stars → **negative**
  - 3 stars → **neutral**
  - 4–5 stars → **positive**
- Train a **Logistic Regression** classifier on TF-IDF features
- Metric: reported with confusion matrix and precision/recall


## Visualization Plan

- **Theme frequency bar chart**: how many reviews fall into each cluster
- **Sentiment distribution pie chart**: share of positive/neutral/negative reviews 
- **Theme/ sentiment heatmap**: which complaint themes correlate with the most negative sentiment
- **Rating distribution over time**: line chart of average number rating by month


## Test Plan

- 80/20 train-test split for the classification model
- Evaluation metrics: accuracy, precision, recall, F1-score, confusion matrix
- Cross-validation to confirm model stability
- Clustering evaluated with silhouette score across k = 3 to 10

  
---

## Project Timeline (8 Weeks)

| Week | Task |
|------|------|
| 1 | Scrape 5,000 Robinhood App Store reviews using app-store-scraper and explore rating distribution and review lengths |
| 2 | Clean data, remove duplicates, non-English reviews, and missing text and normalize text  |
| 3 | Feature engineering: apply TF-IDF vectorization |
| 4 | Run K-Means clustering and manually label each cluster |
| 5 | Refine clusters and generate per-cluster top-term word clouds and frequency charts |
| 6 | Train Logistic Regression model with the rating labels then evaluate with confusion matrix |
| 7 | Build theme x sentiment heatmap, finalize all visualizations and write results in the README |
| 8 | Final code cleanup, Github workflow setup and makefile, and presentation recording |

## Stretch goals 
- [ ] Multi-app comparison analysis
- [ ] Real-time monitoring of new App Store reviews
- [ ] Interactive dashboard front-end
- [ ] Advanced topic modeling methods
