# Customer Feedback Intelligence Pipeline
## Overview

The **Customer Feedback Intelligence Pipeline** is a data science system that analyzes large volumes of customer reviews for the **Robinhood** mobile application to extract useful business insights. The pipeline processes raw customer reviews in text format, converts it into numerical features, applies ML techniques to identify recurring themes and measure customer sentiment.

Goal: Successfully identify the top 5 most frequently occurring complaint themes in Robinhood app store reviews, and classify review sentiment (positive/negative/neutral) with at least 75% accuracy, allowing product teams to prioritize which pain points to address.

## Problem Statement

Companies receive thousands of customer reviews, support tickets, and feedback comments. Manually reading and summarizing this feedback is time-consuming and often leads to missed trends, delayed responses to issues.

This project addresses that problem by building a pipeline that:

* Identifies the top 5 recurring complaint themes using unsupervised clustering
* Classifies customer sentiment (positive/negative/neutral)
* Highlights main product pain points

---

## Project Goals

| # | Goal | Metrics |
|---|------|------------------------|
| 1 | Identify the top 5 recurring complaint themes | K-Means clustering with silhouette score > 0.30 if possible |
| 2 | Classify review sentiment (positive / negative / neutral) | Logistic Regression with > 75% accuracy |
| 3 | Rank pain points (a cluster that contains at least 5% of reviews and has a majority of negative reviews) by frequency | Frequency count of reviews per cluster |
| 4 | Distinguish features by rating | Compare most common terms in 1–2 star vs. 4–5 star reviews |

**If data is limited:** a Kaggle review dataset would be used instead (like [Amazon product reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)) using the same pipeline.


## Dataset

### Primary Data Source

-  **App Store reviews for Robinhood** scraped using the [app-store-scraper](https://pypi.org/project/app-store-scraper/) Python library (no API key neded)
  - Target: ~5,000–10,000 English reviews
  - Fields collected: review text, star rating, review date, review title
  - No API key or authentication required
- Amazon product review datasets (Kaggle as fallback)

### Data Collection

Primary method:

* `app-store-scraper` Python library (for App Store data)

```python
from app_store_scraper import AppStore

app = AppStore(country='us', app_name='robinhood-trading-investing', app_id='938003185')
app.review(how_many=5000)
reviews = app.reviews  # list of dictionary with 'review', 'rating', 'date', etc.
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
- **Sentiment distribution pie chart**: share of positive / neutral / negative reviews overall
- **Theme x sentiment heatmap**: which complaint themes correlate with the most negative sentiment
- **Rating distribution over time**: line chart of average star rating by month


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
