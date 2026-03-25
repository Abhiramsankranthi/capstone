import json
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from pathlib import Path
from src.config import load_config, PROJECT_ROOT

def load_and_filter_articles(config=None):
    """
    Step 1: Loader Fix & Step 2: Financial Relevance Filtering.
    Loads single Kaggle JSONL file and filters by category and market keywords.
    """
    if config is None:
        config = load_config()

    raw_file = PROJECT_ROOT / "data" / "raw" / "News_Category_Dataset_v3.json"
    categories = set(config["sentiment"]["finance_categories"])
    
    # Lead-level filtering: Ensure news is actually about the economy/markets
    fin_keywords = ['S&P', 'FED', 'INFLATION', 'STOCK', 'MARKET', 'ECONOMY', 'INTEREST RATE', 'REVENUE']
    all_articles = []
    
    print(f"Reading {raw_file}...")
    if not raw_file.exists():
        print(f"❌ Error: File not found at {raw_file}")
        return pd.DataFrame()

    with open(raw_file, 'r') as f:
        for line in f:
            a = json.loads(line)
            cat = a.get("category", "").upper()
            if cat in categories:
                text = (a.get("headline", "") + " " + a.get("short_description", "")).upper()
                if any(kw in text for kw in fin_keywords):
                    all_articles.append({
                        "date": a["date"],
                        "headline": a.get("headline", ""),
                        "short_description": a.get("short_description", ""),
                    })

    df = pd.DataFrame(all_articles)
    if df.empty:
        print("⚠️ No articles matched the filters.")
        return df

    # Step 3: Deduplication & Cleaning
    initial_count = len(df)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=['headline'])
    
    print(f"--- Data Audit Results ---")
    print(f"Total relevant articles found: {initial_count}")
    print(f"Articles after deduplication: {len(df)}")
    return df

def run_finbert_inference(articles_df, config=None):
    """
    Step 4: FinBERT Implementation.
    Generates raw sentiment scores using the ProsusAI/finbert model[cite: 83].
    """
    if articles_df.empty:
        return articles_df
    
    if config is None:
        config = load_config()

    model_name = config["sentiment"]["finbert_model"]
    batch_size = config["sentiment"]["batch_size"]
    max_len = config["sentiment"]["max_seq_length"]

    device = torch.device("cpu") # Optimized for MacBook Air stability
    print(f"Loading FinBERT on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    texts = (articles_df["headline"] + ". " + articles_df["short_description"]).tolist()
    scores = []

    for i in tqdm(range(0, len(texts), batch_size), desc="FinBERT inference"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).numpy()
        
        # P(positive) - P(negative)
        batch_scores = probs[:, 0] - probs[:, 1]
        scores.extend(batch_scores.tolist())

    articles_df = articles_df.copy()
    articles_df["sentiment_score"] = scores
    return articles_df

def aggregate_daily_sentiment(scored_df):
    """
    Phase 2: Advanced Feature Engineering.
    Translates raw scores into multi-dimensional daily features[cite: 84, 85].
    """
    if scored_df.empty:
        print("⚠️ No data to aggregate.")
        return
        
    # 1. Sort by date for rolling calculations
    scored_df = scored_df.sort_values("date")
    
    # 2. Advanced Aggregation
    # We calculate multiple statistics to capture the 'flavor' of news each day
    daily = scored_df.groupby("date").agg(
        sent_mean=("sentiment_score", "mean"),
        sent_std=("sentiment_score", "std"),      # Measure of 'disagreement'
        sent_max_neg=("sentiment_score", "min"), # Extreme negative sentiment
        sent_article_count=("sentiment_score", "count")
    ).fillna(0)

    # 3. Volume-Weighted Score (Weighting signal by log-frequency)
    # This prevents 'noise' on days with only one or two headlines
    daily['sent_weighted'] = daily['sent_mean'] * np.log1p(daily['sent_article_count'])

    # 4. Sentiment Momentum (5-day rolling change)
    # Market participants react more to the *change* in outlook than the level
    daily['sent_momentum'] = daily['sent_mean'] - daily['sent_mean'].rolling(window=5).mean()

    out_path = PROJECT_ROOT / "data" / "interim" / "sentiment_daily.parquet"
    daily.to_parquet(out_path)
    
    print(f"✅ Success! Advanced features saved to {out_path}")
    print(f"Columns generated: {list(daily.columns)}")
    return daily

if __name__ == "__main__":
    config = load_config()
    articles = load_and_filter_articles(config)
    if not articles.empty:
        scored = run_finbert_inference(articles, config)
        aggregate_daily_sentiment(scored)