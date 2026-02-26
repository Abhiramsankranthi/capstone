import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from src.config import load_config, PROJECT_ROOT


def fit_hmm_with_bic_selection(features_df=None, config=None):
    if config is None:
        config = load_config()

    if features_df is None:
        features_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "features.parquet")

    hmm_cfg = config["hmm"]
    input_cols = hmm_cfg["input_features"]
    n_iter = hmm_cfg["n_iter"]
    n_fits = hmm_cfg["n_fits"]
    seed = hmm_cfg["random_state"]

    # Prepare input: drop NaN rows
    X_raw = features_df[input_cols].dropna()
    valid_index = X_raw.index

    # Standardize with expanding-window z-scores for production,
    # but for initial exploration use full-sample standardization
    X = (X_raw - X_raw.mean()) / X_raw.std()
    X_arr = X.values

    results = {}
    for n_states in hmm_cfg["n_states_candidates"]:
        best_score = -np.inf
        best_model = None

        for i in range(n_fits):
            model = GaussianHMM(
                n_components=n_states,
                covariance_type="full",
                n_iter=n_iter,
                random_state=seed + i,
            )
            try:
                model.fit(X_arr)
                score = model.score(X_arr)
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception as e:
                print(f"  n_states={n_states}, fit {i}: failed ({e})")

        # BIC = -2 * log_likelihood + n_params * log(n_samples)
        n_params = (
            n_states * n_states  # transition matrix
            + n_states * len(input_cols)  # means
            + n_states * len(input_cols) * (len(input_cols) + 1) // 2  # covariance (full)
            + n_states - 1  # initial state probs
        )
        n_samples = len(X_arr)
        bic = -2 * best_score + n_params * np.log(n_samples)

        results[n_states] = {
            "model": best_model,
            "log_likelihood": best_score,
            "bic": bic,
            "n_params": n_params,
        }
        print(f"n_states={n_states}: LL={best_score:.2f}, BIC={bic:.2f}, n_params={n_params}")

    # Select best by BIC (lowest)
    best_n = min(results, key=lambda k: results[k]["bic"])
    best_model = results[best_n]["model"]
    print(f"\nBest model: {best_n} states (BIC={results[best_n]['bic']:.2f})")

    # Decode states
    states = best_model.predict(X_arr)

    # Label regimes by mean return (state with highest mean return = Bull)
    state_means = pd.DataFrame(
        best_model.means_, columns=input_cols
    )
    # Sort by volatility ascending (low vol = bull, high vol = bear)
    vol_col = [c for c in input_cols if "vol" in c.lower()]
    if vol_col:
        sort_col = vol_col[0]
    else:
        sort_col = input_cols[0]

    state_order = state_means[sort_col].argsort().values
    label_names = ["Bull", "Normal", "Bear/Crisis", "Extreme"][:best_n]
    state_to_label = {state_order[i]: label_names[i] for i in range(best_n)}

    regime_labels = pd.Series(
        [state_to_label[s] for s in states],
        index=valid_index,
        name="regime",
    )

    # Summary stats per regime
    summary = features_df.loc[valid_index].copy()
    summary["regime"] = regime_labels
    regime_stats = summary.groupby("regime").agg(
        mean_return=("sp500_log_return", "mean"),
        std_return=("sp500_log_return", "std"),
        mean_vol=("sp500_realized_vol_21d", lambda x: x.mean()),
        count=("sp500_log_return", "count"),
    )
    regime_stats["pct_of_total"] = regime_stats["count"] / regime_stats["count"].sum() * 100
    print(f"\nRegime statistics:\n{regime_stats}")

    return {
        "best_n_states": best_n,
        "best_model": best_model,
        "all_results": results,
        "regime_labels": regime_labels,
        "regime_stats": regime_stats,
        "valid_index": valid_index,
    }


if __name__ == "__main__":
    fit_hmm_with_bic_selection()
