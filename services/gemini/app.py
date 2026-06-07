from fastapi import FastAPI
import os, requests, json

app = FastAPI(title="LLM Analysis Service")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"


def ask_groq(title: str) -> dict:
    if not GROQ_API_KEY:
        return {
            "verdict":     "unknown",
            "explanation": "Groq API key not configured",
            "confidence":  "low",
            "red_flags":   []
        }

    prompt = f"""You are a fake news detector specialized in cryptocurrency.
Analyze this news title and determine if it's likely FAKE NEWS or REAL NEWS.

Title: "{title}"

Respond in JSON format:
{{
  "verdict": "FAKE" or "REAL",
  "confidence": "high", "medium", or "low",
  "explanation": "brief explanation in 1-2 sentences",
  "red_flags": ["list", "of", "suspicious", "elements"]
}}

Only respond with the JSON, no extra text."""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json"
    }
    payload = {
        "model":       GROQ_MODEL,
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens":  256
    }

    try:
        r    = requests.post(GROQ_URL, headers=headers, json=payload, timeout=15)
        resp = r.json()
        if "error" in resp:
            return {
                "verdict":     "unknown",
                "explanation": resp["error"].get("message", "API error"),
                "confidence":  "low",
                "red_flags":   []
            }
        text = resp["choices"][0]["message"]["content"]
        text = text.strip().strip("```json").strip("```").strip()
        return json.loads(text)
    except Exception as e:
        return {
            "verdict":     "unknown",
            "explanation": f"Error: {str(e)}",
            "confidence":  "low",
            "red_flags":   []
        }


@app.get("/")
def root():
    return {
        "service":      "llm-analysis",
        "model":        GROQ_MODEL,
        "status":       "ok",
        "api_key_set":  bool(GROQ_API_KEY)
    }

@app.post("/analyze")
def analyze(body: dict):
    title = body.get("text", "")
    return ask_groq(title)
