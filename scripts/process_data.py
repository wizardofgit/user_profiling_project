from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from pandas import DataFrame
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load pre-trained RoBERTa model and tokenizer for sentiment analysis
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

def build_tweets_df(raw_tweets: list) -> pd.DataFrame:
    """Convert raw tweet list of JSON data into a structured DataFrame."""
    rows = []
    for t in raw_tweets:
        author = t.get('author', {}) or {}
        created_str = t.get('createdAt')
        ts = pd.to_datetime(created_str, utc=True, errors='coerce')
        created_unix = int(ts.value // 10**9) if pd.notna(ts) else None

        rows.append({
            'tweet_id': t.get('id'),
            'username': author.get('userName'),
            'full_text': t.get('fullText') or t.get('text'),
            'location': author.get('location'),
            'created_at_unix': created_unix,
            'like_count': t.get('likeCount'),
            'view_count': t.get('viewCount'),
        })
    return pd.DataFrame(rows)

def compute_vader_sentiment(text: str) -> float:
    """Compute VADER sentiment compound score for the given text."""
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)["compound"]

def compute_roberta_sentiment(text: str) -> float:
    """Compute RoBERTa sentiment label and score for the given text."""

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Run model
    with torch.no_grad():
        logits = model(**inputs).logits

    # Convert logits → softmax → probabilities
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    labels = ["negative", "neutral", "positive"]
    result = dict(zip(labels, probs))

    compound_score = result["positive"] - result["negative"]

    return compound_score

def extract_tweets_from_json(file_path: Path) -> pd.DataFrame:
    """Extract tweets from a JSON file and return as a DataFrame."""
    with file_path.open("r", encoding="utf-8") as f:
        tweets_json: list[dict[str, Any]] = json.load(f)

    # get rid of unnecessary fields and convert to DataFrame
    tweets: DataFrame = build_tweets_df(tweets_json)

    return tweets

def insert_tweets_into_db(user: pd.DataFrame):
    """Insert tweets into the tweets table."""
    conn = sqlite3.connect("../database/sentiment.db")
    for _, row in user.iterrows():
        conn.execute(
            """
            INSERT INTO tweets (
                id, user_id, timestamp, text, location,
                like_count, view_count,
                vader_compound, roberta_compound
            ) VALUES (
                ?, ?, ?, ?, ?,
                ?, ?,
                ?, ?
            )
            ON CONFLICT(id) DO UPDATE SET
                user_id = excluded.user_id,
                timestamp = excluded.timestamp,
                text = excluded.text,
                location = excluded.location,
                like_count = excluded.like_count,
                view_count = excluded.view_count,
                vader_compound = excluded.vader_compound,
                roberta_compound = excluded.roberta_compound
            """,
            (
                row['tweet_id'],
                row['anonymized_username'],
                row['created_at_unix'],
                row['full_text'],
                row['location'],
                row['like_count'],
                row['view_count'],
                row['vader_compound'],
                row['roberta_compound']
            ),
        )

        conn.commit()

    conn.close()

def insert_users_into_db(usernames: list[str]):
    """Insert unique usernames into the users table."""
    conn = sqlite3.connect("../database/sentiment.db")
    for username in usernames:
        conn.execute(
            "INSERT OR IGNORE INTO users (username) VALUES (?)",
            (username,),
        )

        conn.commit()

    conn.close()

def update_user_profiles():
    """Update user profiles with average sentiment scores and appropriate final label."""
    conn = sqlite3.connect("../database/sentiment.db")
    cursor = conn.execute("SELECT id FROM users")
    user_ids = [row[0] for row in cursor.fetchall()]

    for user_id in user_ids:
        cursor = conn.execute(
            """
            SELECT AVG(vader_compound), AVG(roberta_compound)
            FROM tweets
            WHERE user_id = ?
            """,
            (user_id,),
        )
        avg_vader, avg_roberta = cursor.fetchone()

        compound_sentiment = (avg_vader + avg_roberta) / 2 if avg_vader is not None and avg_roberta is not None else None

        if compound_sentiment > 0.05:
            label = "positive"
        elif compound_sentiment < -0.05:
            label = "negative"
        else:
            label = "neutral"

        conn.execute(
            """
            INSERT INTO profiles (user_id, avg_vader, avg_roberta, compound_sentiment, label)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                avg_vader=excluded.avg_vader,
                avg_roberta=excluded.avg_roberta,
                compound_sentiment=excluded.compound_sentiment,
                label=excluded.label
            """,
            (user_id, avg_vader, avg_roberta, compound_sentiment, label),
        )

        conn.commit()

    conn.close()

def anonymize_user(username: str) -> int:
    """Anonymize username by returning its user ID from the database."""
    conn = sqlite3.connect("../database/sentiment.db")
    cursor = conn.execute(
        "SELECT id FROM users WHERE username = ?",
        (username,),
    )
    user_id = cursor.fetchone()[0]
    conn.close()
    return user_id

def start_processing():
    print("Starting data processing...")

    # process each JSON file
    for file in Path("../data/").glob("*.json"):
        print(f"Processing file: {file.name}")
        
        tweets: pd.DataFrame = extract_tweets_from_json(file)
        
        insert_users_into_db([str(username) for username in tweets['username'].unique()])
        
        tweets['vader_compound'] = tweets['full_text'].apply(compute_vader_sentiment)
        tweets['roberta_compound'] = tweets['full_text'].apply(compute_roberta_sentiment)
        tweets['anonymized_username'] = tweets['username'].apply(anonymize_user)
        insert_tweets_into_db(tweets)
        update_user_profiles()

        print(f"Processing complete for file: {file.name}")

    print("Data processing completed.")

if __name__ == "__main__":
    start_processing()