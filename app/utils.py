def pretty_print_result(sentiment_result, generated_text):
    return f"Sentiment detected: {sentiment_result['label']} (score={sentiment_result['score']:.2f})\n\nGenerated text:\n{generated_text}"
