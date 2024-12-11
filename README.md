
# CSC 487 - Deep Learning: Article Analysis

This repository contains the final project for CSC 487 - Deep Learning, which analyzes sentiment trends in political and current events articles using Natural Language Processing (NLP) techniques.

## Features
* **Sentiment Analysis**: Determines the emotional tone (positive, neutral, negative) of news articles.
* **Web Scraping**: Collects article text from URLs using `BeautifulSoup`.
* **Interactive Dashboard**: Built with `Streamlit` for visualization of sentiment trends.
* **Data Storage**: MongoDB integration for saving and retrieving analyzed articles.

## Future Scope
Implementing bias detection to highlight framing and political leanings in articles.

### Requirements
Ensure the following dependencies are installed:

* `Python 3.7+`
* `streamlit`
* `spacy`
* `nltk`
* `tensorflow`
* `pymongo`
* `beautifulsoup4`
* `matplotlib`

To install the required Python libraries, run:

```
pip install -r requirements.txt
```

Additionally, download the `en_core_web_sm` model for `spaCy`:

```
python -m spacy download en_core_web_sm
```

### How to Run
1. Clone the repository:
   ```
   git clone https://github.com/TarjMecwan/CSC487-Final-Project.git
   ```

2. Navigate to the project directory:
   ```
   cd your-repo-name
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download the `en_core_web_sm` model for `spaCy`:
   ```
   python -m spacy download en_core_web_sm
   ```

5. Start the Streamlit application:
   ```
   streamlit run finalProject.py
   ```

6. Open the provided URL in your browser to interact with the dashboard.

### Usage
#### Input Options
* **Paste Text**: Manually paste article text into the input box for analysis.
* **URL Input**: Provide a URL to fetch and analyze the article content.
* **PDF Upload**: Upload a single PDF file containing the text you want to analyze.
* **Multiple PDF Uploads**: Upload multiple PDF files for batch analysis of their content.
* **MP3 Upload**: Upload an MP3 audio file, which will be transcribed and analyzed for sentiment, emotion, and intent.

#### Output
* **Sentiment Analysis**: Identifies the emotional tone (positive, neutral, negative) of the content.
* **Emotion Mapping**: Generates a radar chart to visualize emotions such as sadness, anger, joy, etc., present in the content.
* **Bias Detection**: Determines whether the content exhibits potential bias.
* **Intent Analysis**: Predicts the intent of the content (e.g., informative, persuasive, entertaining).
* **Summary Generation**: Provides a concise, human-readable summary of the analyzed content.
* **Sentiment Heatmap**: Visualizes the intensity of sentiment across different segments of the text.

### Project Structure
```
.
├── finalProject.py         # Main application script
├── requirements.txt        # Python library dependencies
├── README.md               # Project documentation
```
