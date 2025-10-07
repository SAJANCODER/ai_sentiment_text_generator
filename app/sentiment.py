from transformers import pipeline
from typing import Dict

class SentimentClassifier:
    def __init__(self, model_name: str):
        # use the transformers pipeline
        self.pipe = pipeline("sentiment-analysis", model=model_name, return_all_scores=True)

    def predict(self, text: str) -> Dict:
        """
        Returns a dict: {label: 'positive'|'negative'|'neutral', score: float, raw: pipeline_output}
        We map model outputs to include a neutral threshold.
        """
        results = self.pipe(text)[0]   # list of dicts, ex: [{'label':'POSITIVE','score':0.99}, {...}]
        # Normalize labels
        positive = next((r for r in results if r['label'].lower().startswith("pos")), None)
        negative = next((r for r in results if r['label'].lower().startswith("neg")), None)

        pos_score = positive['score'] if positive else 0.0
        neg_score = negative['score'] if negative else 0.0

        # Determine neutral: if both under threshold or scores close
        if abs(pos_score - neg_score) < 0.12 and max(pos_score, neg_score) < 0.7:
            label = "neutral"
            score = max(pos_score, neg_score)
        else:
            if pos_score >= neg_score:
                label = "positive"
                score = pos_score
            else:
                label = "negative"
                score = neg_score

        return {"label": label, "score": float(score), "raw": results}
