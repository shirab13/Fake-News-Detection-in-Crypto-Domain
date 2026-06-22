from fastapi import FastAPI
import os, requests, json

app = FastAPI(title="LLM Analysis Service")

PROXY_URL = "http://172.21.240.1:8888"


def ask_groq(title: str) -> dict:
    try:
        r = requests.post(f"{PROXY_URL}", json={"text": title}, timeout=20)
        return r.json()
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
        "service": "llm-analysis",
        "status":  "ok",
        "proxy":   PROXY_URL
    }

@app.post("/analyze")
def analyze(body: dict):
    title = body.get("text", "")
    return ask_groq(title)
