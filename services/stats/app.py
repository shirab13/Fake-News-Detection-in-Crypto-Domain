from fastapi import FastAPI
import re

app = FastAPI(title="Stats Service")


def compute_stats(text: str) -> dict:
    words  = text.split()
    n      = len(words) if words else 1

    caps_words = [w for w in words if w.isupper() and len(w) > 1]
    numbers    = re.findall(r'\d+', text)

    return {
        "word_count":          len(words),
        "char_count":          len(text),
        "avg_word_length":     round(sum(len(w) for w in words) / n, 2),
        "caps_word_count":     len(caps_words),
        "caps_ratio":          round(len(caps_words) / n, 4),
        "caps_words":          caps_words[:10],
        "exclamation_count":   text.count('!'),
        "question_count":      text.count('?'),
        "number_count":        len(numbers),
        "has_url":             bool(re.search(r'http[s]?://', text)),
        "special_char_count":  sum(1 for c in text if not c.isalnum() and c != ' '),
        "clickbait_signals": {
            "all_caps":        len(caps_words) > 2,
            "multiple_excl":   text.count('!') > 1,
            "number_in_title": len(numbers) > 0,
            "question_hook":   text.count('?') > 0,
        }
    }


@app.get("/")
def root():
    return {"service": "stats", "status": "ok"}

@app.post("/analyze")
def analyze(body: dict):
    text = body.get("text", "")
    return compute_stats(text)

@app.post("/analyze/batch")
def analyze_batch(body: dict):
    texts = body.get("texts", [])
    return {"results": [compute_stats(t) for t in texts]}
