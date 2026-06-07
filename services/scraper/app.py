from fastapi import FastAPI, BackgroundTasks
from apscheduler.schedulers.background import BackgroundScheduler
import requests, os, psycopg2, feedparser
from datetime import datetime

app = FastAPI(title="Scraper Service")

DB_URL           = os.getenv("DB_URL")
PREPROCESSOR_URL = os.getenv("PREPROCESSOR_URL", "http://preprocessor:8001")
REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT    = os.getenv("REDDIT_USER_AGENT", "FakeNewsBot/1.0")

SUBREDDITS    = ["CryptoCurrency", "Bitcoin", "Buttcoin", "CryptoMoonShots",
                 "SatoshiStreetBets", "CryptoScams", "altcoin", "Superstonk"]

RSS_FEEDS = {
    "CoinDesk":        "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "CoinTelegraph":   "https://cointelegraph.com/rss",
    "BitcoinMagazine": "https://bitcoinmagazine.com/feed",
    "Decrypt":         "https://decrypt.co/feed",
}
MODEL_URL     = os.getenv("MODEL_URL",     "http://model:8002")
SENTIMENT_URL = os.getenv("SENTIMENT_URL", "http://sentiment:8004")
STATS_URL     = os.getenv("STATS_URL",     "http://stats:8003")
TOPIC_URL     = os.getenv("TOPIC_URL",     "http://topic:8005")


def get_db():
    return psycopg2.connect(DB_URL)


def get_reddit_token():
    auth = requests.auth.HTTPBasicAuth(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)
    data = {"grant_type": "client_credentials"}
    headers = {"User-Agent": REDDIT_USER_AGENT}
    r = requests.post("https://www.reddit.com/api/v1/access_token",
                      auth=auth, data=data, headers=headers)
    return r.json().get("access_token")


def scrape_subreddit(subreddit: str, limit: int = 100):
    headers = {"User-Agent": REDDIT_USER_AGENT}

    if REDDIT_CLIENT_ID and REDDIT_CLIENT_ID != "your_client_id_here":
        token = get_reddit_token()
        headers["Authorization"] = f"bearer {token}"
        url = f"https://oauth.reddit.com/r/{subreddit}/hot.json?limit={limit}"
    else:
        url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"

    r = requests.get(url, headers=headers)
    posts = r.json().get("data", {}).get("children", [])

    results = []
    for p in posts:
        d = p["data"]
        results.append({
            "title":        d.get("title", ""),
            "text":         d.get("selftext", "")[:500],
            "url":          d.get("url", ""),
            "subreddit":    subreddit,
            "score":        d.get("score", 0),
            "upvote_ratio": d.get("upvote_ratio", 0.5),
            "num_comments": d.get("num_comments", 0),
        })
    return results


def enrich_post(text: str) -> dict:
    """Call all analysis services for a single text. Returns enrichment dict."""
    enrichment = {
        "label_pred": 1, "label_prob": 0.5,
        "sentiment_polarity": 0.0, "sentiment_subjectivity": 0.5,
        "word_count": 0, "caps_ratio": 0.0, "exclamation_count": 0,
        "topic": None,
    }

    # Preprocess
    clean = text
    try:
        r = requests.post(f"{PREPROCESSOR_URL}/preprocess", json={"text": text}, timeout=5)
        clean = r.json().get("cleaned", text)
    except Exception:
        pass

    # Model prediction: LSTM probability with strict threshold
    # prob > 0.7 → REAL, otherwise FAKE (conservative: uncertain = suspicious)
    try:
        mr = requests.post(f"{MODEL_URL}/predict", json={"text": clean, "model": "lstm"}, timeout=10)
        prob = mr.json().get("probability", 0.5)
        enrichment["label_pred"] = 1 if prob > 0.7 else 0
        enrichment["label_prob"] = prob
    except Exception:
        pass

    # Sentiment
    try:
        sr = requests.post(f"{SENTIMENT_URL}/analyze", json={"text": clean}, timeout=5)
        tb = sr.json().get("textblob", {})
        enrichment["sentiment_polarity"]    = tb.get("polarity", 0.0)
        enrichment["sentiment_subjectivity"] = tb.get("subjectivity", 0.5)
    except Exception:
        pass

    # Stats
    try:
        st = requests.post(f"{STATS_URL}/analyze", json={"text": text}, timeout=5)
        sd = st.json()
        enrichment["word_count"]        = sd.get("word_count", 0)
        enrichment["caps_ratio"]        = sd.get("caps_ratio", 0.0)
        enrichment["exclamation_count"] = sd.get("exclamation_count", 0)
    except Exception:
        pass

    # Topic
    try:
        tr = requests.post(f"{TOPIC_URL}/analyze", json={"text": text}, timeout=5)
        enrichment["topic"] = tr.json().get("primary_topic")
    except Exception:
        pass

    return enrichment


def process_and_save(posts: list):
    conn = get_db()
    cur  = conn.cursor()

    for post in posts:
        try:
            e = enrich_post(post["title"])
            cur.execute("""
                INSERT INTO posts (title, text, url, subreddit, score, upvote_ratio, num_comments,
                                   label_pred, label_prob, word_count, caps_ratio, exclamation_count,
                                   sentiment_polarity, sentiment_subjectivity, topic)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (post["title"], post["text"], post["url"],
                  post["subreddit"], post["score"], post["upvote_ratio"], post["num_comments"],
                  e["label_pred"], e["label_prob"], e["word_count"], e["caps_ratio"],
                  e["exclamation_count"], e["sentiment_polarity"], e["sentiment_subjectivity"],
                  e["topic"]))
            conn.commit()
        except Exception as ex:
            print(f"Error processing post: {ex}")

    cur.close()
    conn.close()


def batch_process_existing():
    """Update existing posts that have NULL label_pred."""
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("SELECT id, title FROM posts WHERE label_pred IS NULL")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    print(f"[batch] Found {len(rows)} posts to enrich...")
    conn = get_db()
    cur  = conn.cursor()

    for post_id, title in rows:
        try:
            e = enrich_post(title)
            cur.execute("""
                UPDATE posts SET
                    label_pred            = %s,
                    label_prob            = %s,
                    word_count            = %s,
                    caps_ratio            = %s,
                    exclamation_count     = %s,
                    sentiment_polarity    = %s,
                    sentiment_subjectivity= %s,
                    topic                 = %s
                WHERE id = %s
            """, (e["label_pred"], e["label_prob"], e["word_count"], e["caps_ratio"],
                  e["exclamation_count"], e["sentiment_polarity"], e["sentiment_subjectivity"],
                  e["topic"], post_id))
            conn.commit()
        except Exception as ex:
            print(f"[batch] Error on post {post_id}: {ex}")

    cur.close()
    conn.close()
    print("[batch] Done enriching existing posts.")


def scrape_all():
    print(f"[{datetime.now()}] Starting scheduled scrape...")
    all_posts = []
    for sub in SUBREDDITS:
        try:
            posts = scrape_subreddit(sub, limit=100)
            all_posts.extend(posts)
            print(f"  {sub}: {len(posts)} posts")
        except Exception as e:
            print(f"  Error scraping {sub}: {e}")
    process_and_save(all_posts)
    print(f"[{datetime.now()}] Done. Total: {len(all_posts)} posts")


def scrape_rss():
    print(f"[{datetime.now()}] Starting RSS scrape...")
    all_posts = []
    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:30]:
                title = entry.get("title", "").strip()
                link  = entry.get("link", "")
                if not title:
                    continue
                all_posts.append({
                    "title":        title,
                    "text":         entry.get("summary", "")[:500],
                    "url":          link,
                    "subreddit":    source,
                    "score":        0,
                    "upvote_ratio": 0.5,
                    "num_comments": 0,
                })
            print(f"  {source}: {len(feed.entries[:30])} articles")
        except Exception as e:
            print(f"  Error scraping {source}: {e}")
    process_and_save(all_posts)
    print(f"[{datetime.now()}] RSS done. Total: {len(all_posts)} articles")


# ─── Scheduler ───────────────────────────────────────────
scheduler = BackgroundScheduler()
scheduler.add_job(scrape_all, 'interval', hours=24)
scheduler.add_job(scrape_rss,  'interval', hours=24)
scheduler.start()


# ─── API Endpoints ────────────────────────────────────────
@app.get("/")
def root():
    return {"service": "scraper", "status": "running"}

@app.post("/scrape/now")
def scrape_now(background_tasks: BackgroundTasks):
    background_tasks.add_task(scrape_all)
    return {"message": "Scraping started in background"}

@app.post("/scrape/rss")
def scrape_rss_now(background_tasks: BackgroundTasks):
    background_tasks.add_task(scrape_rss)
    return {"message": "RSS scraping started in background", "sources": list(RSS_FEEDS.keys())}

@app.post("/scrape/url")
def scrape_url(url: str):
    try:
        json_url = url.rstrip("/") + ".json"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
        }
        # use OAuth if credentials are available
        if REDDIT_CLIENT_ID and REDDIT_CLIENT_ID != "your_client_id_here":
            token = get_reddit_token()
            headers["Authorization"] = f"bearer {token}"
            headers["User-Agent"] = REDDIT_USER_AGENT
            json_url = url.rstrip("/").replace("www.reddit.com", "oauth.reddit.com") + ".json"

        r = requests.get(json_url, headers=headers, timeout=10)

        if r.status_code != 200:
            # fallback: extract title from URL slug
            parts = [p for p in url.rstrip("/").split("/") if p]
            if len(parts) >= 2:
                slug = parts[-1]
                title_from_url = slug.replace("_", " ").replace("-", " ").title()
                post = {
                    "title": title_from_url, "text": "", "url": url,
                    "subreddit": parts[4] if len(parts) > 4 else "",
                    "score": 0, "upvote_ratio": 0.5, "num_comments": 0,
                }
                process_and_save([post])
                return {"message": "Title extracted from URL (no credentials)", "title": title_from_url}
            return {"error": f"Reddit returned HTTP {r.status_code}. Try using Reddit credentials."}

        raw = r.json()
        data = raw[0]["data"]["children"][0]["data"]
        post = {
            "title":        data.get("title", ""),
            "text":         data.get("selftext", "")[:500],
            "url":          url,
            "subreddit":    data.get("subreddit", ""),
            "score":        data.get("score", 0),
            "upvote_ratio": data.get("upvote_ratio", 0.5),
            "num_comments": data.get("num_comments", 0),
        }
        process_and_save([post])
        return {"message": "Post scraped and saved", "title": post["title"]}
    except Exception as e:
        return {"error": str(e)}

@app.post("/process-existing")
def process_existing(background_tasks: BackgroundTasks):
    """Enrich existing posts that have NULL label_pred with model/sentiment/topic data."""
    background_tasks.add_task(batch_process_existing)
    return {"message": "Batch processing started in background"}

@app.get("/stats")
def scraper_stats():
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("SELECT COUNT(*), subreddit FROM posts GROUP BY subreddit")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {"counts_by_subreddit": {r[1]: r[0] for r in rows}}
