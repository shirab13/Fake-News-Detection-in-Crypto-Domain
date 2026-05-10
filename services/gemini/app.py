from fastapi import FastAPI
import os, requests, json

app = FastAPI(title="Gemini Service")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_URL     = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"


def ask_gemini(title: str) -> dict:
    if not GEMINI_API_KEY:
        return {
            "verdict":     "unknown",
            "explanation": "Gemini API key not configured",
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

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 256}
    }

    try:
        r = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            json=payload,
            timeout=15
        )
        resp = r.json()
        print(f"Gemini API response: {resp}")
        if "error" in resp:
            return {
                "verdict":     "unknown",
                "explanation": resp["error"].get("message", "API error"),
                "confidence":  "low",
                "red_flags":   []
            }
        text = resp["candidates"][0]["content"]["parts"][0]["text"]
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
        "service": "gemini",
        "status":  "ok",
        "api_key_set": bool(GEMINI_API_KEY)
    }

@app.post("/analyze")
def analyze(body: dict):
    title = body.get("text", "")
    return ask_gemini(title)
