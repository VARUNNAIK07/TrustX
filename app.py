from __future__ import annotations

from pathlib import Path
import json
import re

import pandas as pd
from flask import Flask, jsonify, request

try:
    import fasttext
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise SystemExit(
        "fasttext is not installed. Install it with: pip install fasttext"
    ) from exc

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise SystemExit(
        "xgboost is not installed. Install it with: pip install xgboost"
    ) from exc


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "fasttext_xgboost_hybrid_output"
FASTTEXT_MODEL_PATH = ARTIFACT_DIR / "model.bin"
XGBOOST_MODEL_PATH = ARTIFACT_DIR / "xgboost_model.json"
PREPROCESSING_PATH = ARTIFACT_DIR / "preprocessing.json"

LABELS = {0: "REAL", 1: "FAKE"}

app = Flask(__name__)

fasttext_model: fasttext.FastText._FastText | None = None
xgboost_model: XGBClassifier | None = None
preprocessing_config: dict[str, object] | None = None


def clean_text(text: str) -> str:
    cleaned = re.sub(r"@\w+", " ", str(text or ""))
    cleaned = re.sub(r"http\S+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def parse_count(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().lower().replace(",", "")
    if not text:
        return 0.0

    multiplier = 1.0
    if text.endswith("k"):
        multiplier = 1_000.0
        text = text[:-1]
    elif text.endswith("m"):
        multiplier = 1_000_000.0
        text = text[:-1]

    try:
        return float(text) * multiplier
    except ValueError:
        return 0.0


def load_artifacts() -> None:
    global fasttext_model, xgboost_model, preprocessing_config

    if not FASTTEXT_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing FastText model: {FASTTEXT_MODEL_PATH}")
    if not XGBOOST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing XGBoost model: {XGBOOST_MODEL_PATH}")
    if not PREPROCESSING_PATH.exists():
        raise FileNotFoundError(f"Missing preprocessing config: {PREPROCESSING_PATH}")

    fasttext_model = fasttext.load_model(str(FASTTEXT_MODEL_PATH))

    xgboost_model = XGBClassifier()
    xgboost_model.load_model(str(XGBOOST_MODEL_PATH))

    preprocessing_config = json.loads(PREPROCESSING_PATH.read_text(encoding="utf-8"))


def ensure_models_loaded() -> None:
    if fasttext_model is None or xgboost_model is None or preprocessing_config is None:
        load_artifacts()


def get_fasttext_probability(text: str) -> float:
    assert fasttext_model is not None
    labels, probabilities = fasttext_model.predict(clean_text(text), k=2)
    probability_map = dict(zip(labels, probabilities))
    return float(probability_map.get("__label__1", 0.0))


def build_numeric_features(payload: dict[str, object], text: str) -> dict[str, float]:
    likes = parse_count(payload.get("likes"))
    reposts = max(parse_count(payload.get("retweets")), 1.0)
    replies = parse_count(payload.get("replies"))

    alpha_chars = [char for char in text if char.isalpha()]
    uppercase_chars = [char for char in alpha_chars if char.isupper()]

    return {
        "like_rt_ratio": likes / reposts if reposts else 0.0,
        "reply_rt_ratio": replies / reposts if reposts else 0.0,
        "word_count": float(len(re.findall(r"\b\w+\b", text))),
        "caps_ratio": float(len(uppercase_chars) / len(alpha_chars)) if alpha_chars else 0.0,
        "mentions": float(len(re.findall(r"@\w+", text))),
        "URLs": float(len(re.findall(r"https?://\S+|www\.\S+", text))),
        "has_hashtag": float(bool(re.search(r"#\w+", text))),
        "has_exclamation": float("!" in text),
    }


def build_feature_frame(payload: dict[str, object]) -> tuple[pd.DataFrame, float]:
    ensure_models_loaded()
    assert preprocessing_config is not None

    raw_text = str(payload.get("text", "")).strip()
    if not raw_text:
        raise ValueError("Text is required.")

    fasttext_prob = get_fasttext_probability(raw_text)
    numeric_features = build_numeric_features(payload, raw_text)
    feature_values = {**numeric_features, "fasttext_prob": fasttext_prob}

    medians = preprocessing_config.get("feature_medians", {})
    ordered_features = preprocessing_config.get("hybrid_feature_columns", [])

    row = {}
    for feature_name in ordered_features:
        if feature_name in feature_values:
            row[feature_name] = float(feature_values[feature_name])
        else:
            row[feature_name] = float(medians.get(feature_name, 0.0))

    return pd.DataFrame([row]), fasttext_prob


def build_response(payload: dict[str, object]) -> dict[str, object]:
    ensure_models_loaded()
    assert xgboost_model is not None

    features, fasttext_prob = build_feature_frame(payload)
    fake_probability = float(xgboost_model.predict_proba(features)[0][1])
    prediction = 1 if fake_probability >= 0.5 else 0

    reasons = []
    if fasttext_prob >= 0.7:
        reasons.append("FastText flagged suspicious language")
    if features.at[0, "caps_ratio"] >= 0.35:
        reasons.append("High uppercase usage")
    if features.at[0, "has_exclamation"] >= 1.0:
        reasons.append("Contains emphatic punctuation")
    if features.at[0, "has_hashtag"] >= 1.0:
        reasons.append("Contains hashtags")
    if not reasons:
        reasons.append("Model relied on overall hybrid score")

    return {
        "prediction": LABELS[prediction],
        "confidence": round(fake_probability * 100 if prediction == 1 else (1 - fake_probability) * 100, 2),
        "fake_probability": round(fake_probability, 4),
        "fasttext_probability": round(fasttext_prob, 4),
        "reasons": reasons,
    }


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/health", methods=["GET"])
def health() -> tuple[object, int]:
    try:
        ensure_models_loaded()
        return jsonify({"status": "ok", "artifacts_dir": str(ARTIFACT_DIR)}), 200
    except Exception as exc:  # pragma: no cover - runtime visibility
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict() -> tuple[object, int]:
    if request.method == "OPTIONS":
        return jsonify({"ok": True}), 200

    payload = request.get_json(silent=True) or {}
    try:
        result = build_response(payload)
        return jsonify(result), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - runtime visibility
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    ensure_models_loaded()
    app.run(host="127.0.0.1", port=5000, debug=False)
