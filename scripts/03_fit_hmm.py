"""Fit HMM for regime detection and validate results."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.config import load_config, PROJECT_ROOT
from src.models.hmm import fit_hmm_with_bic_selection
from src.validation.regime_validation import validate_regimes


def main():
    config = load_config()

    print("\n=== Fitting HMM ===")
    features_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "features.parquet")
    results = fit_hmm_with_bic_selection(features_df, config)

    print("\n=== Validating Regimes ===")
    validate_regimes(features_df, results["regime_labels"], config)

    # Save regime labels
    results["regime_labels"].to_frame().to_parquet(
        PROJECT_ROOT / "data" / "processed" / "regime_labels.parquet"
    )
    print("\nRegime labels saved to data/processed/regime_labels.parquet")
    print("\n=== HMM Pipeline Complete ===")


if __name__ == "__main__":
    main()
