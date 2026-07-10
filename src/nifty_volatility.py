import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model


def download_and_calculate_volatility():
    print("--- Step 1: Downloading NIFTY 50 Data ---")
    # Download NIFTY 50 data (symbol: ^NSEI)
    nifty = yf.download("^NSEI", start="2003-01-01", end="2023-12-31")
    nifty.reset_index(inplace=True)

    # Save raw downloaded data
    raw_output = "data/NIFTY50_2003_2023.csv"
    nifty.to_csv(raw_output, index=False)
    print(f"Raw data saved to {raw_output}")

    print("\n--- Step 2: Processing Returns & GARCH Modeling ---")
    df = pd.read_csv(raw_output)
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)

    # Ensure Close is numeric
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    # Compute daily log returns
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))

    # Drop missing values and scale returns to % for GARCH convergence stability
    returns = df["LogReturn"].dropna() * 100

    # Fit GARCH(1,1) model
    model = arch_model(returns, vol="GARCH", p=1, q=1)
    res = model.fit(disp="off")
    print(res.summary())

    # Map conditional volatility back to the main dataframe
    df = df.loc[returns.index].copy()
    df["GARCH_Volatility"] = res.conditional_volatility

    # Save output to data directory
    vol_output_filename = "data/nifty_vol_sentiment_garch.csv"
    df.to_csv(vol_output_filename, index=False)
    print(f"\nVolatility output successfully saved to {vol_output_filename}")


if __name__ == "__main__":
    # Create data directory if it doesn't exist
    import os

    os.makedirs("data", exist_ok=True)
    download_and_calculate_volatility()
