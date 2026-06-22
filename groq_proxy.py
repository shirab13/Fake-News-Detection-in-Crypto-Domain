from http.server import HTTPServer, BaseHTTPRequestHandler
import json, urllib.request, os

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"


def ask_groq(title):
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

    payload = json.dumps({
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 256
    }).encode()

    req = urllib.request.Request(
        GROQ_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0"
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        text = data["choices"][0]["message"]["content"]
        text = text.strip().strip("```json").strip("```").strip()
        return json.loads(text)
    except Exception as e:
        return {"verdict": "unknown", "confidence": "low", "explanation": str(e), "red_flags": []}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = json.loads(self.rfile.read(length)) if length else {}
        result = ask_groq(body.get("text", ""))
        out    = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    print("GROQ Proxy running on port 8888 - Docker containers can reach it at http://172.19.0.1:8888")
    HTTPServer(("0.0.0.0", 8888), Handler).serve_forever()
