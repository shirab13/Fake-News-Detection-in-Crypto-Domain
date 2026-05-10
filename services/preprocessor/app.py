from fastapi import FastAPI
import re, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('punkt',     quiet=True)

app        = FastAPI(title="Preprocessor Service")
lemmatizer = WordNetLemmatizer()
STOPS      = set(stopwords.words('english'))


def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\d+', 'NUM', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [lemmatizer.lemmatize(w) for w in text.split()
              if w not in STOPS and len(w) > 2]
    return ' '.join(tokens)


@app.get("/")
def root():
    return {"service": "preprocessor", "status": "ok"}

@app.post("/preprocess")
def preprocess(body: dict):
    text = body.get("text", "")
    return {
        "original": text,
        "cleaned":  clean(text)
    }

@app.post("/preprocess/batch")
def preprocess_batch(body: dict):
    texts = body.get("texts", [])
    return {"cleaned": [clean(t) for t in texts]}
