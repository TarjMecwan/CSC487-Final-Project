import streamlit as st
import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import re
from pymongo import MongoClient
from transformers import pipeline
import PyPDF2
import requests
from bs4 import BeautifulSoup
from pydub import AudioSegment
import os
import numpy as np

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize NLTK sentiment analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client['sentiment_analysis']
collection = db['articles']

# Hugging Face models
bias_classifier = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

# Streamlit setup
st.title("Comprehensive Article Analyzer")
st.sidebar.title("Upload or Paste Content")

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop])

# Fetch article content from URL
def fetch_article(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return " ".join([p.text for p in soup.find_all("p")])

# PDF extraction function
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# MP3 extraction function
def extract_text_from_mp3(file):
    temp_mp3 = "temp.mp3"
    with open(temp_mp3, "wb") as f:
        f.write(file.read())
    audio = AudioSegment.from_file(temp_mp3)
    audio.export("temp.wav", format="wav")
    os.remove(temp_mp3)

    import whisper
    model = whisper.load_model("base")
    result = model.transcribe("temp.wav")
    os.remove("temp.wav")
    return result["text"]

# Summarization function
def summarize_text(text):
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]["summary_text"]

# Sentiment analysis using NLTK
def analyze_sentiment_nltk(text):
    scores = sia.polarity_scores(text)
    return scores

# Political bias analysis (updated)
def analyze_bias(text):
    result = bias_classifier(text[:512])  # Limit to 512 characters for processing
    label = result[0]['label']
    score = result[0]['score']
    
    # Consider 'Center' as not biased, everything else as biased
    if label == "Center":
        return {"bias": "Not Biased", "confidence": score}
    else:
        return {"bias": "Biased", "confidence": score}

# Emotion analysis
def analyze_emotions(text):
    emotion_scores = emotion_model(text[:512])  # Limit to 512 characters for processing
    scores = {item["label"]: item["score"] for item in emotion_scores[0]}
    return scores

# Visualization for emotions (Radar Chart)
def visualize_emotions(emotion_scores):
    labels = list(emotion_scores.keys())
    values = list(emotion_scores.values())
    values += values[:1]  # Close the radar chart loop

    angles = np.linspace(0, 2 * np.pi, len(labels) + 1, endpoint=True)
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='blue', alpha=0.25)
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.set_yticks([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title("Emotion Analysis Radar Chart")
    st.pyplot(fig)

# Visualization for political bias (updated)
def visualize_bias(bias_result):
    labels = ["Not Biased", "Biased"]
    values = [0, 0]
    
    if bias_result["bias"] == "Not Biased":
        values[0] = bias_result["confidence"]
    else:
        values[1] = bias_result["confidence"]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=["green", "orange"])
    plt.xlabel("Bias Category")
    plt.ylabel("Confidence")
    plt.title("Bias Analysis")
    st.pyplot(plt)

# Visualization for sentiment analysis (Bar Chart)
def visualize_sentiment(sentiment_scores):
    categories = list(sentiment_scores.keys())
    values = list(sentiment_scores.values())
    plt.figure(figsize=(6, 4))
    plt.bar(categories, values)
    plt.xlabel('Sentiment Categories')
    plt.ylabel('Scores')
    plt.title('Sentiment Analysis')
    st.pyplot(plt)

# Input section
input_type = st.sidebar.radio("Input Method", ["Paste Text", "URL", "Upload PDF", "Upload MP3", "Upload Multiple PDFs"])

if input_type == "Paste Text":
    user_input = st.sidebar.text_area("Paste your article here:")
elif input_type == "URL":
    url = st.sidebar.text_input("Enter the URL of the article:")
    if url:
        user_input = fetch_article(url)
        st.sidebar.write("Fetched Article:")
        st.sidebar.write(user_input)
elif input_type == "Upload PDF":
    uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type="pdf")
    if uploaded_pdf:
        user_input = extract_text_from_pdf(uploaded_pdf)
        st.sidebar.write("Extracted Text:")
        st.sidebar.write(user_input)
elif input_type == "Upload MP3":
    uploaded_mp3 = st.sidebar.file_uploader("Upload an MP3", type="mp3")
    if uploaded_mp3:
        user_input = extract_text_from_mp3(uploaded_mp3)
        st.sidebar.write("Transcribed Text:")
        st.sidebar.write(user_input)
elif input_type == "Upload Multiple PDFs":
    uploaded_pdfs = st.sidebar.file_uploader("Upload Multiple PDFs", type="pdf", accept_multiple_files=True)
    if uploaded_pdfs:
        user_input = "\n".join([extract_text_from_pdf(pdf) for pdf in uploaded_pdfs])
        st.sidebar.write("Extracted Text from Uploaded PDFs:")
        st.sidebar.write(user_input)

if st.sidebar.button("Analyze"):
    if user_input:
        preprocessed_text = preprocess_text(user_input)
        sentiment_scores = analyze_sentiment_nltk(preprocessed_text)
        bias_result = analyze_bias(preprocessed_text)
        emotion_scores = analyze_emotions(preprocessed_text)
        summary = summarize_text(preprocessed_text)

        # Save to MongoDB
        collection.insert_one({
            "original_text": user_input,
            "preprocessed_text": preprocessed_text,
            "sentiment_scores": sentiment_scores,
            "bias_result": bias_result,
            "emotion_scores": emotion_scores,
            "summary": summary,
        })

        # Display results
        st.write("### Sentiment Analysis Results")
        st.write(sentiment_scores)
        visualize_sentiment(sentiment_scores)

        st.write("### Bias Analysis Results")
        st.write(f"Bias: {bias_result['bias']} with {bias_result['confidence']:.2f} confidence")
        visualize_bias(bias_result)

        st.write("### Emotion Analysis Results")
        st.write(emotion_scores)
        visualize_emotions(emotion_scores)

        st.write("### Summary")
        st.write(summary)
    else:
        st.sidebar.error("Please provide text, a URL, or upload a file.")
