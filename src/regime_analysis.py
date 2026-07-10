import os
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


def run_regime_analysis():
    print("--- Step 6: Running Markov Regime-Switching Analysis ---")

    input_file = "data/merged_nifty_sentiment.csv"
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Missing base merged file: '{input_file}'. Run data_merging.py first."
        )

    # Load data and guarantee chronological indexing
    df = pd.read_csv(input_file)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Clean target variable matrix
    df = df.dropna(subset=["garch_volatility"])
    y = df["garch_volatility"]

    # Fit Markov Regression model (2 states: Calm/Low Vol vs Turbulent/High Vol)
    print("Fitting Markov model parameters via EM...")
    markov_model = MarkovRegression(
        endog=y, k_regimes=2, trend="c", switching_variance=True
    )
    markov_results = markov_model.fit(em_iter=10, search_reps=20)
    print(markov_results.summary())

    # Extract smoothed marginal probabilities for State 1 (High Volatility)
    df["regime_prob"] = markov_results.smoothed_marginal_probabilities[1]

    # Assign binary threshold classification rule (alpha = 0.50)
    df["predicted_regime"] = (df["regime_prob"] > 0.5).astype(int)

    # Display descriptive state density mapping
    print("\nEmpirical state density partition:")
    print(df["predicted_regime"].value_counts(normalize=True))

    # Save finalized dataset containing features and regimes
    output_file = "data/final_dataset_with_regimes.csv"
    df.to_csv(output_file, index=False)
    print(f"\nUnified analytical framework saved to -> {output_file}")


if __name__ == "__main__":
    run_regime_analysis()
