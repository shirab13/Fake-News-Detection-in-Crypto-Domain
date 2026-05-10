from fastapi import FastAPI
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app      = FastAPI(title="Sentiment Service")
analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(text: str) -> dict:
    # TextBlob
    blob = TextBlob(text)
    tb_polarity    = round(blob.sentiment.polarity, 4)
    tb_subjectivity= round(blob.sentiment.subjectivity, 4)

    # VADER
    vader = analyzer.polarity_scores(text)

    # label טקסטואלי
    if tb_polarity > 0.1:
        sentiment_label = "positive"
    elif tb_polarity < -0.1:
        sentiment_label = "negative"
    else:
        sentiment_label = "neutral"

    return {
        "textblob": {
            "polarity":     tb_polarity,
            "subjectivity": tb_subjectivity,
            "sentiment":    sentiment_label,
        },
        "vader": {
            "positive": round(vader["pos"], 4),
            "negative": round(vader["neg"], 4),
            "neutral":  round(vader["neu"], 4),
            "compound": round(vader["compound"], 4),
            "sentiment": "positive" if vader["compound"] >= 0.05
                         else "negative" if vader["compound"] <= -0.05
                         else "neutral"
        },
        "subjectivity_label": "subjective" if tb_subjectivity > 0.5 else "objective"
    }


@app.get("/")
def root():
    return {"service": "sentiment", "status": "ok"}

@app.post("/analyze")
def analyze(body: dict):
    text = body.get("text", "")
    return analyze_sentiment(text)

@app.post("/analyze/batch")
def analyze_batch(body: dict):
    texts = body.get("texts", [])
    return {"results": [analyze_sentiment(t) for t in texts]}
