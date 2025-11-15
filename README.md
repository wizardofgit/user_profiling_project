# user_profiling_project
The project utilizes a simple system for profilizing users and their posts via sentiment analysis to provide features for predicting LoL matches outcomes.

# 📄 Database Schema (`sentiment.db`)

All project data is stored in a **single SQLite database**:

```
./database/sentiment.db
```

The database contains three main tables: **users**, **tweets**, and **profiles**.

---

## **1. `users`**
Stores unique Twitter/X users discovered during data collection, that have been anonymized.

**Columns:**
- `id` — INTEGER, primary key, autoincrement
- `username` — TEXT, unique, not null

**DDL:**
```sql
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL
);
```

---

## **2. `tweets`**
Stores tweet content, metadata, and sentiment scores.

**Columns:**
- `id` — INTEGER, primary key
- `user_id` — INTEGER, foreign key → users(id), not null
- `timestamp` — TEXT, ISO-8601 timestamp, not null
- `text` — TEXT, tweet content, not null
- `location` — TEXT, user location (nullable)
- `like_count` — INTEGER, number of likes (nullable)
- `view_count` — INTEGER, number of views (nullable)
- `vader_compound` — REAL, VADER sentiment score (nullable)
- `roberta_compound` — REAL, RoBERTa sentiment score (nullable)

**DDL:**
```sql
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
```

---

## **3. `profiles`**
Stores aggregated sentiment per user.

**Columns:**
- `user_id` — INTEGER, primary key, foreign key → users(id)
- `avg_vader` — REAL, average VADER score (nullable)
- `avg_roberta` — REAL, average RoBERTa score (nullable)
- `compound_sentiment` — REAL, combined metric (nullable)
- `label` — TEXT, sentiment classification label (nullable)

**DDL:**
```sql
CREATE TABLE IF NOT EXISTS profiles (
    user_id INTEGER PRIMARY KEY,
    avg_vader REAL,
    avg_roberta REAL,
    compound_sentiment REAL,
    label TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```