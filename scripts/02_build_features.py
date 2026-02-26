"""Build all features and merge into a single dataset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.features.merge import merge_all_features


def main():
    config = load_config()

    print("\n=== Building Features & Merging ===")
    merged = merge_all_features(config)

    print(f"\nDataset shape: {merged.shape}")
    print(f"Date range: {merged.index.min()} to {merged.index.max()}")
    print(f"Missing values:\n{merged.isnull().sum()}")
    print("\n=== Feature Build Complete ===")


if __name__ == "__main__":
    main()
