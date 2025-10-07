# AI Sentiment Aware Text Generator  

### Developed by: D. Sajan
*An intelligent system that generates essays or paragraphs aligned with the sentiment of user input — positive, negative, or neutral.*

---

## 🌟 Overview

This project implements an **AI Text Generator** that analyzes the **sentiment** of a user-provided prompt and generates text that matches that emotional tone.  
For example:
- A **positive** prompt yields an uplifting, encouraging essay.  
- A **negative** prompt produces a critical or pessimistic response.  
- A **neutral** prompt returns a balanced, objective explanation.  

The project integrates **sentiment analysis** and **text generation** using **pre-trained transformer models** from the Hugging Face ecosystem, wrapped with a modern **Streamlit** frontend for seamless user interaction.

---

## Features

1.	Automatic sentiment detection using a fine-tuned DistilBERT model.  
2.	Sentiment-aligned paragraph generation using GPT-2 / GPT-Neo.  
3.	Manual sentiment override for user control.  
4.	Adjustable output length and creativity (temperature).  
5.	Interactive Streamlit-based web interface.  
6.	Optional fine-tuning scripts for custom datasets.  
7.	Lightweight Docker-ready deployment setup.  

---

## 🏗️ System Architecture

**1. Sentiment Detection Layer:**  
Analyzes user input using a pre-trained classifier (`distilbert-base-uncased-finetuned-sst-2-english`) and identifies the sentiment (positive, negative, neutral).  

**2. Text Generation Layer:**  
Uses a language model (`gpt2-medium` or `EleutherAI/gpt-neo-125M`) to generate paragraphs conditioned on both the user prompt and detected sentiment.  

**3. Control & Prompt Engineering Layer:**  
Adds sentiment-specific instructions to steer the generator effectively (e.g., *"Write a positive paragraph about..."*).  

**4. Frontend Layer:**  
A Streamlit interface where users can input text, view detected sentiment, override it, and control text length and creativity.  

**5. (Optional) Fine-Tuning Module:**  
Includes scripts for improving the generator and sentiment classifier using custom datasets for better domain alignment.  

---




