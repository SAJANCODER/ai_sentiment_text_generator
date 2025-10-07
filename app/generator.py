# app/generator.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from config import GEN_MODEL, DEVICE, DEFAULT_MAX_TOKENS

class TextGenerator:
    def __init__(self, model_name: str = GEN_MODEL, device: str = DEVICE):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)
        self.device = device

    def craft_prompt(self, user_prompt: str, sentiment: str) -> str:
        instructions = {
            "positive": "Write a positive, uplifting paragraph about the following prompt:",
            "negative": "Write a critical or negative paragraph about the following prompt:",
            "neutral" : "Write a neutral, balanced paragraph about the following prompt:"
        }
        return f"{instructions.get(sentiment, instructions['neutral'])}\n\n{user_prompt}\n\nParagraph:"

    def generate(self, user_prompt: str, sentiment: str, max_tokens: int = DEFAULT_MAX_TOKENS, temperature: float = 0.8):
        prompt = self.craft_prompt(user_prompt, sentiment)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.92,
            top_k=50,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=3
        )
        gen_text = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        return gen_text.strip()
