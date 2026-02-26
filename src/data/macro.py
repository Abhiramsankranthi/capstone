import pandas as pd
from fredapi import Fred
from src.config import load_config, get_fred_api_key, PROJECT_ROOT


def fetch_macro_data(config=None):
    if config is None:
        config = load_config()

    fred = Fred(api_key=get_fred_api_key())
    start = config["date_range"]["start"]
    end = config["date_range"]["end"]
    series_ids = list(config["fred"]["series"].keys())

    frames = {}
    for sid in series_ids:
        print(f"Fetching FRED series: {sid}")
        s = fred.get_series(sid, observation_start=start, observation_end=end)
        frames[sid] = s

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    # Resample to daily and forward-fill (monthly/weekly → daily)
    df = df.resample("D").first().ffill()

    out_path = PROJECT_ROOT / "data" / "interim" / "macro_indicators.parquet"
    df.to_parquet(out_path)
    print(f"Saved macro indicators: {df.shape} to {out_path}")
    return df


if __name__ == "__main__":
    fetch_macro_data()
