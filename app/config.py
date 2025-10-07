SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
# Generation model - pick small for demo, medium for quality.
GEN_MODEL = "gpt2-medium"  # or "EleutherAI/gpt-neo-125M" for open weights
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
DEFAULT_MAX_TOKENS = 150
