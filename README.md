# NIFTY 50 Volatility & FinBERT Sentiment Analysis

A quantitative data science pipeline analyzing the temporal relationship between financial news sentiment and stock market volatility on the National Stock Exchange of India (NSE). 

This project tests the **Sentiment-Driven Volatility Hypothesis** against the **Reactive Media Hypothesis** across distinct market regimes from 2003 to 2023.

---

##  Methodology

* **Sentiment Extraction:** Uses Hugging Face's `ProsusAI/FinBERT` to evaluate daily financial news sentiment.
* **Volatility Modeling:** Estimates daily conditional market variance using a GARCH(1,1) model.
* **Regime Classification:** Applies a 2-State Markov Regime-Switching model to classify the market into Low Volatility (Calm) and High Volatility (Turbulent) states.
* **Causality Testing:** Runs Granger Causality tests across multiple lag horizons to determine directional lead-lag relationships.

---

##  Key Findings

The statistical testing strongly supports the **Reactive Media Hypothesis**:
* **Volatility Leads Sentiment:** Historical market volatility strongly drives subsequent shifts in financial news tone (p < 0.001).
* **Sentiment Does Not Lead Volatility:** Daily financial news sentiment fails to Granger-cause market volatility (p = 0.519).
* **Regime Invariance:** This reactive media behavior remains consistent whether the market is in a calm state or experiencing high-stress turbulence.

---

##  Key Visualizations 
*Note: All output graphs are generated and saved in the `data/` directory.*

* **sentiment_vs_volatility.png**: Scatter plot showing the spatial density separations between low and high market volatility regimes.
* **markov_switching_regimes.png**: Time-series plot with high-stress market states dynamically shaded in red.
* **granger_Pvalues_across_lags.png**: Line chart tracking p-values across daily lags, mathematically validating the final hypothesis.

---

##  How to Run

1. Install dependencies:
   `pip install -r requirements.txt`

2. Execute the pipeline scripts sequentially from the `src/` folder:
   * `nifty_volatility.py` -> `sentiment_extraction.py` -> `data_merging.py`
   * `regime_analysis.py` -> `causality_testing.py` -> `generate_plots.py`
