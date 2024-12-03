import streamlit as st
import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pymongo import MongoClient
import matplotlib.pyplot as plt
import re
from bs4 import BeautifulSoup
import requests
import os
import PyPDF2
from pathlib import Path

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize NLTK sentiment analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client['sentiment_analysis']
collection = db['articles']

# Streamlit setup
st.title("Political and Current Events Article Sentiment Analyzer")
st.sidebar.title("Upload or Paste Article Text")

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

# Folder extraction function
def extract_text_from_folder(folder):
    text = ""
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder, file)
            with open(file_path, "rb") as pdf_file:
                text += extract_text_from_pdf(pdf_file)
    return text

# Sentiment analysis using NLTK
def analyze_sentiment_nltk(text):
    scores = sia.polarity_scores(text)
    return scores

# Visualization
def visualize_sentiment(sentiment_scores):
    categories = list(sentiment_scores.keys())
    values = list(sentiment_scores.values())
    plt.bar(categories, values)
    plt.xlabel('Sentiment Categories')
    plt.ylabel('Scores')
    plt.title('Sentiment Analysis')
    st.pyplot(plt)

# Input section
input_type = st.sidebar.radio("Input Method", ["Paste Text", "URL", "Upload PDF", "Upload Folder of PDFs"])

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
elif input_type == "Upload Folder of PDFs":
    folder_path = st.sidebar.text_input("Enter the folder path containing PDFs:")
    if folder_path and Path(folder_path).exists():
        user_input = extract_text_from_folder(folder_path)
        st.sidebar.write("Extracted Text:")
        st.sidebar.write(user_input)
    else:
        st.sidebar.error("Folder not found. Please check the path.")

if st.sidebar.button("Analyze"):
    if user_input:
        preprocessed_text = preprocess_text(user_input)
        sentiment_scores = analyze_sentiment_nltk(preprocessed_text)
        
        # Save to MongoDB
        collection.insert_one({
            "original_text": user_input,
            "preprocessed_text": preprocessed_text,
            "sentiment_scores": sentiment_scores
        })
        
        # Display results
        st.write("### Sentiment Analysis Results")
        st.write(sentiment_scores)
        visualize_sentiment(sentiment_scores)
    else:
        st.sidebar.error("Please provide text, a URL, or upload a PDF.")

# Show historical trends
if st.checkbox("Show Historical Trends"):
    all_articles = list(collection.find())
    st.write(f"Number of Analyzed Articles: {len(all_articles)}")
    sentiment_trends = {"positive": 0, "neutral": 0, "negative": 0}
    
    for article in all_articles:
        sentiment = article['sentiment_scores']
        sentiment_trends["positive"] += sentiment['pos']
        sentiment_trends["neutral"] += sentiment['neu']
        sentiment_trends["negative"] += sentiment['neg']
    
    plt.bar(sentiment_trends.keys(), sentiment_trends.values())
    plt.title("Historical Sentiment Trends")
    plt.xlabel("Sentiment Type")
    plt.ylabel("Aggregate Score")
    st.pyplot(plt)
