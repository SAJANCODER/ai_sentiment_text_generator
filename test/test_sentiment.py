from app.sentiment import SentimentClassifier
def test_sentiment_basic():
    clf = SentimentClassifier("distilbert-base-uncased-finetuned-sst-2-english")
    r = clf.predict("I love this product, it's amazing!")
    assert r["label"] in ["positive", "neutral", "negative"]
