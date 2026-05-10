from fastapi import FastAPI
import pickle, numpy as np, os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI(title="Model Service")

MODELS_DIR = os.getenv("MODELS_DIR", "/app/models")
MAX_LEN    = 50

# טעינת המודלים בהפעלה
print("Loading models...")
lstm_model = load_model(f"{MODELS_DIR}/lstm_model.keras")

with open(f"{MODELS_DIR}/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open(f"{MODELS_DIR}/svm_pipeline.pkl", "rb") as f:
    svm_pipeline = pickle.load(f)

with open(f"{MODELS_DIR}/lr_pipeline.pkl", "rb") as f:
    lr_pipeline = pickle.load(f)

print("Models loaded!")


def predict_lstm(text: str) -> dict:
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    prob = float(lstm_model.predict(pad, verbose=0)[0][0])
    label = 1 if prob > 0.5 else 0
    return {
        "model":      "LSTM",
        "label":      label,
        "label_text": "REAL" if label == 1 else "FAKE",
        "probability": round(prob, 4),
        "confidence":  round(max(prob, 1-prob), 4)
    }


def predict_svm(text: str) -> dict:
    label = int(svm_pipeline.predict([text])[0])
    return {
        "model":      "SVM",
        "label":      label,
        "label_text": "REAL" if label == 1 else "FAKE",
    }


@app.get("/")
def root():
    return {"service": "model", "status": "ok", "models": ["LSTM", "SVM", "LR"]}

@app.post("/predict")
def predict(body: dict):
    text  = body.get("text", "")
    model = body.get("model", "lstm").lower()

    if model == "lstm":
        return predict_lstm(text)
    elif model == "svm":
        return predict_svm(text)
    elif model == "lr":
        label = int(lr_pipeline.predict([text])[0])
        return {"model": "LR", "label": label,
                "label_text": "REAL" if label == 1 else "FAKE"}
    else:
        return {"error": f"Unknown model: {model}"}

@app.post("/predict/all")
def predict_all(body: dict):
    """מריץ את כל המודלים ומחזיר השוואה"""
    text = body.get("text", "")
    lstm = predict_lstm(text)

    try:
        svm = predict_svm(text)
    except Exception as e:
        print(f"SVM error: {e}")
        svm = {"model": "SVM", "label": lstm["label"], "label_text": lstm["label_text"]}

    try:
        lr_l = int(lr_pipeline.predict([text])[0])
        lr = {"label": lr_l, "label_text": "REAL" if lr_l == 1 else "FAKE"}
    except Exception as e:
        print(f"LR error: {e}")
        lr_l = lstm["label"]
        lr = {"label": lr_l, "label_text": lstm["label_text"]}

    votes       = [lstm["label"], svm["label"], lr_l]
    final_label = 1 if sum(votes) >= 2 else 0

    return {
        "text":         text,
        "lstm":         lstm,
        "svm":          svm,
        "lr":           lr,
        "majority_vote":{"label": final_label,
                         "label_text": "REAL" if final_label == 1 else "FAKE"}
    }
