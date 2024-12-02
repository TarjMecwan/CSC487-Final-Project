
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
   git clone https://github.com/your-username/your-repo-name.git
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

#### Output
* Sentiment scores are displayed as a bar chart.
* The analysis is saved to MongoDB for historical trend tracking.

### Project Structure
```
.
├── finalProject.py         # Main application script
├── requirements.txt        # Python library dependencies
├── README.md               # Project documentation
└── data/                   # (Optional) Pre-scraped or example datasets
```
