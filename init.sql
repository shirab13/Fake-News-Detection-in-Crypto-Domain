CREATE TABLE IF NOT EXISTS posts (
    id              SERIAL PRIMARY KEY,
    title           TEXT NOT NULL,
    text            TEXT,
    url             TEXT,
    subreddit       TEXT,
    score           INTEGER,
    upvote_ratio    FLOAT,
    num_comments    INTEGER,
    created_at      TIMESTAMP DEFAULT NOW(),
    scraped_at      TIMESTAMP DEFAULT NOW(),

    -- תוצאות ניתוח
    label_pred      INTEGER,          -- 0=FAKE, 1=REAL
    label_prob      FLOAT,            -- הסתברות
    word_count      INTEGER,
    caps_ratio      FLOAT,
    exclamation_count INTEGER,
    sentiment_polarity    FLOAT,
    sentiment_subjectivity FLOAT,
    topic           TEXT,
    gemini_verdict  TEXT
);

CREATE INDEX IF NOT EXISTS idx_posts_label ON posts(label_pred);
CREATE INDEX IF NOT EXISTS idx_posts_subreddit ON posts(subreddit);
CREATE INDEX IF NOT EXISTS idx_posts_created ON posts(created_at);
