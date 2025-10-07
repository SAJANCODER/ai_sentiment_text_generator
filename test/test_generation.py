from app.generator import TextGenerator
def test_generate_short():
    gen = TextGenerator("gpt2")
    out = gen.generate("Climate change and its impacts", "neutral", max_tokens=50)
    assert isinstance(out, str) and len(out) > 10
