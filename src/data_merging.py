import pandas as pd


def merge_market_and_sentiment_data():
    print("--- Step 5: Merging Datasets ---")

    nifty_file = "data/nifty_vol_sentiment_garch.csv"
    sentiment_file = "data/daily_sentiment_finbert.csv"

    # Read tracking datasets
    nifty = pd.read_csv(nifty_file)
    sentiment = pd.read_csv(sentiment_file)

    # Standardize column structures
    nifty.columns = nifty.columns.str.strip().str.lower()
    sentiment.columns = sentiment.columns.str.strip().str.lower()

    if "date" not in nifty.columns and "date" in nifty.columns:
        nifty.rename(columns={"Date": "date"}, inplace=True)

    nifty["date"] = pd.to_datetime(nifty["date"])
    sentiment["date"] = pd.to_datetime(sentiment["date"])

    # Synchronize datasets via an inner timeline merge
    merged_df = pd.merge(nifty, sentiment, on="date", how="inner")

    # Safe placeholder structure for 'predicted_regime'
    # This keeps our data structure clean before applying the Markov logic
    if "garch_volatility" in merged_df.columns:
        merged_df["predicted_regime"] = 0

    # Explicit column list matching project requirements
    cols_to_keep = [
        "date",
        "logreturn",
        "garch_volatility",
        "daily_sentiment_mean",
        "daily_sentiment_std",
        "sentiment_rolling_mean",
        "article_count",
    ]

    if "predicted_regime" in merged_df.columns:
        cols_to_keep.append("predicted_regime")

    # Filter out only existing matching keys safely
    existing_cols = [col for col in cols_to_keep if col in merged_df.columns]
    final_df = merged_df[existing_cols]

    # Chronologically align the combined sequence
    final_df = final_df.sort_values("date").reset_index(drop=True)

    # Save output compilation
    output_filename = "data/merged_nifty_sentiment.csv"
    final_df.to_csv(output_filename, index=False)

    print("\nMerged dataset pipeline complete!")
    print(final_df.head())
    print(f"Total merged tracking records: {len(final_df)}")


if __name__ == "__main__":
    import os

    # Validation fallback check
    if os.path.exists("data/nifty_vol_sentiment_garch.csv") and os.path.exists(
        "data/daily_sentiment_finbert.csv"
    ):
        merge_market_and_sentiment_data()
    else:
        print(
            "Error: Pre-requisite files missing. Make sure to run 'nifty_volatility.py' and 'sentiment_extraction.py' first."
        )
