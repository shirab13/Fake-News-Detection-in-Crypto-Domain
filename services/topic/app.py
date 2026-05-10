from fastapi import FastAPI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

app = FastAPI(title="Topic Analysis Service")

# Topics קריפטו מוגדרים מראש
CRYPTO_TOPICS = {
    0: "Price & Market",
    1: "Technology & Development",
    2: "Regulation & Government",
    3: "Scam & Fraud",
    4: "Mining & Energy",
    5: "DeFi & NFT",
    6: "Adoption & News",
    7: "Community & Social",
}

# Keywords לכל topic - לזיהוי ישיר
TOPIC_KEYWORDS = {
    "Price & Market":          ["price", "pump", "dump", "bull", "bear", "crash", "moon", "ath", "market", "trade"],
    "Technology & Development":["blockchain", "protocol", "upgrade", "fork", "layer", "smart", "contract", "eth", "btc"],
    "Regulation & Government": ["sec", "government", "ban", "law", "regulation", "legal", "tax", "cbdc", "policy"],
    "Scam & Fraud":            ["scam", "fraud", "hack", "steal", "rug", "pull", "ponzi", "fake", "warning"],
    "Mining & Energy":         ["mine", "mining", "miner", "energy", "hash", "power", "electricity", "gpu"],
    "DeFi & NFT":              ["defi", "nft", "yield", "liquidity", "swap", "dao", "token", "wallet"],
    "Adoption & News":         ["adopt", "payment", "company", "partner", "launch", "announce", "accept"],
    "Community & Social":      ["community", "reddit", "twitter", "elon", "influencer", "meme", "social"],
}


def detect_topic(text: str) -> dict:
    text_lower = text.lower()
    scores = {}

    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[topic] = score

    best_topic = max(scores, key=scores.get)
    best_score = scores[best_topic]

    if best_score == 0:
        best_topic = "General Crypto"

    # top 3 topics
    sorted_topics = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top3 = [(t, s) for t, s in sorted_topics[:3] if s > 0]

    return {
        "primary_topic":  best_topic,
        "topic_scores":   scores,
        "top_3_topics":   top3,
        "match_strength": "high" if best_score >= 3 else "medium" if best_score >= 1 else "low"
    }


@app.get("/")
def root():
    return {"service": "topic", "status": "ok", "topics": list(TOPIC_KEYWORDS.keys())}

@app.post("/analyze")
def analyze(body: dict):
    text = body.get("text", "")
    return detect_topic(text)

@app.post("/analyze/batch")
def analyze_batch(body: dict):
    texts = body.get("texts", [])
    return {"results": [detect_topic(t) for t in texts]}
