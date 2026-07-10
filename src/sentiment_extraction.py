import os
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Setup hardware configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using processing device: {device}")

# Load FinBERT tokenizer and model globally
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
model.to(device)

label_names = ["negative", "neutral", "positive"]


def get_finbert_sentiment(text):
    """Tokenizes text string and runs transformer model inference."""
    try:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        sentiment = label_names[torch.argmax(probs)]
        # continuous metric calculated as: positive probability - negative probability
        score = probs[0][2].item() - probs[0][0].item()
        return sentiment, score
    except Exception:
        return "error", 0.0


def process_news_sentiment(input_news_csv):
    print(f"\n--- Step 3: Loading News Dataset from {input_news_csv} ---")
    df = pd.read_csv(input_news_csv)
    print("Columns detected:", df.columns.tolist())

    # Keep relevant baseline columns
    df = df[["Unnamed: 0", "Date", "Title", "Description"]]

    # Combine Title + Description for a richer semantic context footprint
    df["text"] = df["Title"].fillna("") + ". " + df["Description"].fillna("")

    # Clean dates and drop malformed sequences
    df["date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.dropna(subset=["date", "text"], inplace=True)
    print(f"Total valid articles loaded for evaluation: {len(df)}")

    # Execute batch scoring with tracking wrapper
    print("\nRunning FinBERT model inference (this may take a while)...")
    tqdm.pandas()
    df[["sentiment_label", "sentiment_score"]] = df["text"].progress_apply(
        lambda x: pd.Series(get_finbert_sentiment(str(x)))
    )

    # Clean and save fine-grained article-level records
    article_csv = "data/article_level_sentiment.csv"
    df.to_csv(article_csv, index=False)
    print(f"Saved article-level logs -> {article_csv}")

    # --- Step 4: Perform Daily Aggregate Accounting ---
    print("\nAggregating daily metrics...")
    daily_sentiment = (
        df.groupby("date")
        .agg(
            daily_sentiment_mean=("sentiment_score", "mean"),
            daily_sentiment_std=("sentiment_score", "std"),
            article_count=("sentiment_score", "count"),
        )
        .reset_index()
    )

    # Compute a smoothed 5-day rolling mean to minimize high frequency daily variance
    daily_sentiment["sentiment_rolling_mean"] = (
        daily_sentiment["daily_sentiment_mean"].rolling(window=5).mean()
    )

    daily_csv = "data/daily_sentiment_finbert.csv"
    daily_sentiment.to_csv(daily_csv, index=False)
    print(f"Saved daily aggregated sentiment timeline -> {daily_csv}")
    print(daily_sentiment.head())


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    # REPLACE with your actual initial raw Kaggle financial news file name placed in the data/ folder
    raw_news_file = "data/raw_indian_news_dataset.csv"

    if os.path.exists(raw_news_file):
        process_news_sentiment(raw_news_file)
    else:
        print(
            f"Error: Please place your
