import streamlit as st
from config import SENTIMENT_MODEL, GEN_MODEL, DEVICE
from generator import TextGenerator
from sentiment import SentimentClassifier

st.set_page_config(page_title="AI Sentiment Aware Text Generator", layout="centered")

st.title("AI Text Generator Sentiment Aligned Paragraphs")
st.caption("prompt aware, sentiment aligned generation")

# Instantiate models (cache to avoid reloading on rerun)
@st.cache_resource
def load_models():
    sent = SentimentClassifier(SENTIMENT_MODEL)
    gen = TextGenerator(GEN_MODEL)
    return sent, gen

sentiment_model, generator = load_models()

with st.form("generation_form"):
    prompt = st.text_area("Enter your prompt", height=140, placeholder="E.g., 'Describe the impact of remote work on productivity.'")
    manual_override = st.selectbox("Override sentiment (optional)", ["Auto-detect", "positive", "neutral", "negative"])
    length = st.slider("Maximum generation length (words approx.)", min_value=50, max_value=400, value=150, step=10)
    temperature = st.slider("Sampling temperature (creativity)", min_value=0.2, max_value=1.2, value=0.8, step=0.1)
    submitted = st.form_submit_button("Generate")

if submitted:
    if not prompt.strip():
        st.error("Please enter a prompt.")
    else:
        # Sentiment detection
        result = sentiment_model.predict(prompt)
        detected = result["label"]
        if manual_override != "Auto-detect":
            detected = manual_override

        with st.spinner("Generating text..."):
            # convert word count approx to tokens: assume 1.3 tokens per word
            max_tokens = int(length * 1.3)
            generated = generator.generate(prompt, detected, max_tokens=max_tokens, temperature=temperature)

        st.markdown(f"**Sentiment (detected):** {result['label']}  — confidence: {result['score']:.2f}")
        if manual_override != "Auto-detect":
            st.markdown(f"**Manual override used**: {manual_override}")
        st.write("---")
        st.write(generated)
        st.write("---")
        # small quality controls
        st.info("Tip: use Manual override to test stronger alignment. Lower temperature → more deterministic output.")
