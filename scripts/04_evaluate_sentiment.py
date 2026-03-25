import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
from src.config import PROJECT_ROOT

def evaluate_feature_importance():
    # 1. Load the fully merged dataset
    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "features.parquet")
    
    # 2. Focus on the 'Sentiment Era' (2012-2022)
    # This prevents the early 2000s (no news) from skewing the results
    analysis_df = df.dropna(subset=['sent_weighted']).copy()
    
    # 3. Define Features and Target
    # We exclude forward returns from the features to prevent leakage
    target = 'fwd_return_1d'
    features = [c for c in analysis_df.columns if c not in ['fwd_return_1d', 'fwd_return_5d']]
    
    X = analysis_df[features]
    y = analysis_df[target]
    
    # 4. Train a quick LightGBM model
    model = LGBMRegressor(n_estimators=100, random_state=42, importance_type='gain')
    model.fit(X, y)
    
    # 5. Extract and Plot Importance
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(12, 10))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(20), palette='viridis')
    plt.title('Top 20 Features: Incremental Value of Sentiment')
    plt.xlabel('Importance (Gain)')
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "data" / "processed" / "sentiment_importance.png")
    plt.show()

    print("--- Lead NLP Evaluation ---")
    print(f"Total Features Analyzed: {len(features)}")
    sent_cols = [c for c in importance_df['Feature'] if 'sent_' in c]
    for col in sent_cols:
        rank = importance_df[importance_df['Feature'] == col].index[0] + 1
        print(f"Feature: {col} | Rank: {rank}/{len(features)}")

if __name__ == "__main__":
    evaluate_feature_importance()