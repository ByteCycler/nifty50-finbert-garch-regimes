import os
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests


def execute_granger_loop(
    data, sample_label, maxlag=5, cause="daily_sentiment_mean", effect="garch_volatility"
):
    """Executes explicit direction causality testing frameworks."""
    print(
        f"\n🔹 Testing Framework [{sample_label}]: Does '{cause}' Granger-cause '{effect}'?"
    )
    # Target formatting validation requires vector alignment: [Effect, Cause]
    test_data = data[[effect, cause]].dropna()

    try:
        results = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)
        lag_p_values = []
        for lag in range(1, maxlag + 1):
            p_val = results[lag][0]["ssr_chi2test"][1]
            lag_p_values.append(p_val)
            print(f"  Lag {lag} → Chi2 p-value: {p_val:.4f}")

        min_p = min(lag_p_values)
        print(f"  Conclusion: Minimum p-value over horizon is {min_p:.4f}")
        return lag_p_values
    except Exception as e:
        print(f"  Testing processing execution aborted: {e}")
        return [1.0] * maxlag


def run_all_causality_tests():
    print("--- Step 7: Running Directional Econometric Causality Pipelines ---")
    dataset_path = "data/final_dataset_with_regimes.csv"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Pre-requisite dataset missing: '{dataset_path}'."
        )

    df = pd.read_csv(dataset_path)

    # --- Phase A: Full Sample System Verification ---
    print("\n================ FULL SAMPLE ESTIMATION ================")
    # 1. Sentiment -> Volatility
    execute_granger_loop(
        df,
        "Full Sample",
        maxlag=5,
        cause="daily_sentiment_mean",
        effect="garch_volatility",
    )
    # 2. Volatility -> Sentiment
    execute_granger_loop(
        df,
        "Full Sample",
        maxlag=5,
        cause="garch_volatility",
        effect="daily_sentiment_mean",
    )

    # --- Phase B: Stratified System Verification Across Regimes ---
    low_vol_subset = df[df["predicted_regime"] == 0]
    high_vol_subset = df[df["predicted_regime"] == 1]

    print("\n================ LOW VOLATILITY REGIME STAGE ================")
    execute_granger_loop(
        low_vol_subset,
        "Low Vol Regime",
        maxlag=5,
        cause="daily_sentiment_mean",
        effect="garch_volatility",
    )
    execute_granger_loop(
        low_vol_subset,
        "Low Vol Regime",
        maxlag=5,
        cause="garch_volatility",
        effect="daily_sentiment_mean",
    )

    print("\n================ HIGH VOLATILITY REGIME STAGE ================")
    execute_granger_loop(
        high_vol_subset,
        "High Vol Regime",
        maxlag=5,
        cause="daily_sentiment_mean",
        effect="garch_volatility",
    )
    execute_granger_loop(
        high_vol_subset,
        "High Vol Regime",
        maxlag=5,
        cause="garch_volatility",
        effect="daily_sentiment_mean",
    )


if __name__ == "__main__":
    run_all_causality_tests()
