import json
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from src.config import load_config, PROJECT_ROOT


def load_and_filter_articles(config=None):
    if config is None:
        config = load_config()

    raw_dir = PROJECT_ROOT / config["sentiment"]["raw_dir"]
    categories = set(config["sentiment"]["finance_categories"])

    all_articles = []
    for year in range(2012, 2023):
        fpath = raw_dir / f"data_{year}.json"
        if not fpath.exists():
            print(f"Warning: {fpath} not found, skipping")
            continue
        with open(fpath) as f:
            articles = json.load(f)
        for a in articles:
            if a.get("category", "").upper() in categories:
                all_articles.append({
                    "date": a["date"],
                    "headline": a.get("headline", ""),
                    "short_description": a.get("short_description", ""),
                    "category": a.get("category", ""),
                })

    df = pd.DataFrame(all_articles)
    df["date"] = pd.to_datetime(df["date"])
    print(f"Filtered {len(df)} finance-relevant articles from {df['date'].min()} to {df['date'].max()}")
    return df


def run_finbert_inference(articles_df, config=None):
    if config is None:
        config = load_config()

    model_name = config["sentiment"]["finbert_model"]
    batch_size = config["sentiment"]["batch_size"]
    max_len = config["sentiment"]["max_seq_length"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading FinBERT on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # Combine headline + short_description
    texts = (articles_df["headline"] + ". " + articles_df["short_description"]).tolist()

    scores = []
    labels = []
    label_map = {0: "positive", 1: "negative", 2: "neutral"}

    for i in tqdm(range(0, len(texts), batch_size), desc="FinBERT inference"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=max_len, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        # Score = P(positive) - P(negative)
        batch_scores = probs[:, 0] - probs[:, 1]
        batch_labels = [label_map[idx] for idx in probs.argmax(axis=1)]
        scores.extend(batch_scores.tolist())
        labels.extend(batch_labels)

    articles_df = articles_df.copy()
    articles_df["sentiment_score"] = scores
    articles_df["sentiment_label"] = labels
    print(f"FinBERT inference complete. Label distribution:\n{articles_df['sentiment_label'].value_counts()}")
    return articles_df


def aggregate_daily_sentiment(scored_df):
    daily = scored_df.groupby("date").agg(
        sent_mean=("sentiment_score", "mean"),
        sent_max_neg=("sentiment_score", "min"),
        sent_article_count=("sentiment_score", "count"),
    )
    daily.index.name = "date"

    out_path = PROJECT_ROOT / "data" / "interim" / "sentiment_daily.parquet"
    daily.to_parquet(out_path)
    print(f"Saved daily sentiment: {daily.shape} to {out_path}")
    return daily


if __name__ == "__main__":
    config = load_config()
    articles = load_and_filter_articles(config)
    scored = run_finbert_inference(articles, config)
    aggregate_daily_sentiment(scored)
