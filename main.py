from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import joblib
import re

app = FastAPI(
    title="Twitter Sentiment Analysis API",
    description="3-class sentiment classifier on airline tweets — 76.9% accuracy (vs 33% random baseline)",
    version="1.0.0"
)

pipeline = joblib.load("sentiment_pipeline.joblib")

LABEL_MAP  = {0: 'negative', 1: 'neutral', 2: 'positive'}
EMOJI_MAP  = {'negative': 'Negative', 'neutral': 'Neutral', 'positive': 'Positive'}

# ABSA keyword routing
ASPECT_KEYWORDS = {
    'food':    ['food', 'meal', 'drink', 'snack', 'hungry', 'eat', 'lunch', 'dinner', 'breakfast', 'catering'],
    'staff':   ['staff', 'crew', 'attendant', 'service', 'rude', 'friendly', 'helpful', 'employee', 'agent'],
    'luggage': ['bag', 'baggage', 'luggage', 'lost', 'missing', 'suitcase', 'belongings'],
    'delay':   ['delay', 'late', 'wait', 'delayed', 'hours', 'cancelled', 'cancel', 'reschedule'],
    'comfort': ['seat', 'cramped', 'space', 'legroom', 'comfortable', 'uncomfortable', 'narrow'],
    'price':   ['price', 'expensive', 'cheap', 'cost', 'fee', 'charge', 'refund', 'money']
}

DEPARTMENT_MAP = {
    'food':    'Catering',
    'staff':   'HR / Customer Experience',
    'luggage': 'Baggage Operations',
    'delay':   'Operations / Scheduling',
    'comfort': 'Product / Fleet',
    'price':   'Revenue Management'
}


def clean_tweet(text: str) -> str:
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def detect_aspects(text: str) -> List[Dict]:
    text_lower = text.lower()
    detected = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            detected.append(aspect)
    return detected


def predict_absa(text: str) -> List[Dict]:
    aspects = detect_aspects(text)
    results = []
    for aspect in aspects:
        keywords = ASPECT_KEYWORDS[aspect]
        aspect_text = ' '.join([w for w in text.lower().split()
                                 if w in keywords or w in clean_tweet(text).split()])
        combined = f"{aspect} {text}"
        probs  = pipeline.predict_proba([clean_tweet(combined)])[0]
        label  = LABEL_MAP[probs.argmax()]
        conf   = round(float(probs.max()), 3)
        results.append({
            "aspect":     aspect,
            "sentiment":  label,
            "confidence": conf,
            "department": DEPARTMENT_MAP[aspect],
            "action":     "Flag for review" if label == "negative" else
                          "Amplify" if label == "positive" else "Monitor"
        })
    return results


class TweetInput(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "@UnitedAirlines lost my bag AGAIN. Crew was rude and flight was delayed 3 hours!"
            }
        }


class BatchInput(BaseModel):
    tweets: List[str]


@app.get("/")
def root():
    return {
        "message": "Twitter Sentiment Analysis API is live",
        "model": "TF-IDF + Logistic Regression — 76.9% accuracy on 14,640 tweets (vs 33% baseline)",
        "docs": "/docs",
        "version": "1.0"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(tweet: TweetInput):
    try:
        cleaned = clean_tweet(tweet.text)
        probs   = pipeline.predict_proba([cleaned])[0]
        label   = LABEL_MAP[probs.argmax()]
        conf    = round(float(probs.max()), 3)

        neg, neu, pos = round(float(probs[0]), 3), round(float(probs[1]), 3), round(float(probs[2]), 3)

        absa = predict_absa(tweet.text)

        return {
            "text":        tweet.text,
            "sentiment":   label,
            "confidence":  conf,
            "probabilities": {
                "negative": neg,
                "neutral":  neu,
                "positive": pos
            },
            "absa_results": absa,
            "departments_flagged": [r["department"] for r in absa if r["sentiment"] == "negative"],
            "summary": f"{label.upper()} ({conf*100:.1f}% confidence) — {len(absa)} aspect(s) detected"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch")
def predict_batch(batch: BatchInput):
    results = []
    for tweet_text in batch.tweets:
        result = predict(TweetInput(text=tweet_text))
        results.append(result)

    sentiment_counts = {"negative": 0, "neutral": 0, "positive": 0}
    for r in results:
        sentiment_counts[r["sentiment"]] += 1

    all_depts = []
    for r in results:
        all_depts.extend(r["departments_flagged"])
    dept_counts = {}
    for d in all_depts:
        dept_counts[d] = dept_counts.get(d, 0) + 1

    return {
        "predictions":      results,
        "count":            len(results),
        "sentiment_summary": sentiment_counts,
        "departments_to_action": dept_counts
    }