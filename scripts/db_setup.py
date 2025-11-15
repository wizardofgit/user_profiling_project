import sqlite3
import os

# Paths relative to this script:
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "database", "sentiment.db")

# --------- SQL SCHEMAS ---------

USERS_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL
);
"""

TWEETS_SQL = """
CREATE TABLE IF NOT EXISTS tweets (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    text TEXT NOT NULL,
    location TEXT,
    like_count INTEGER,
    view_count INTEGER,
    vader_compound REAL,
    roberta_compound REAL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
"""

PROFILES_SQL = """
CREATE TABLE IF NOT EXISTS profiles (
    user_id INTEGER PRIMARY KEY,
    avg_vader REAL,
    avg_roberta REAL,
    compound_sentiment REAL,
    label TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
"""


# --------- CREATE DB + TABLES ---------

def create_database(debug: bool = False):
    if debug:
        print("Debug mode is ON. Recreating the database.")

        # Remove existing database file if it exists
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)

    # Ensure /database folder exists
    os.makedirs(os.path.join(BASE_DIR, "database"), exist_ok=True)

    print(f"Creating/connecting to database at: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Enable FK enforcement
    cur.execute("PRAGMA foreign_keys = ON;")

    # Create tables
    cur.execute(USERS_SQL)
    cur.execute(TWEETS_SQL)
    cur.execute(PROFILES_SQL)

    conn.commit()
    conn.close()

    print("Database and tables created successfully!")

if __name__ == "__main__":
    create_database(debug=True)