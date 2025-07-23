# Amazon Reviews Sentiment Analyzer 
<img src="https://upload.wikimedia.org/wikipedia/commons/4/41/Antu_amazon-mp3-store-source.svg" alt="Amazon Logo" width="70"/>  <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face" width="80"/><img src="https://upload.wikimedia.org/wikipedia/commons/6/67/Numpy-svgrepo-com.svg" alt="NumPy" width="70"/>  <img src="https://upload.wikimedia.org/wikipedia/commons/8/84/Matplotlib_icon.svg" alt="Matplotlib" width="70"/>  <img src="https://upload.wikimedia.org/wikipedia/commons/4/45/Logo-seaborn.png" alt="Seaborn" width="70"/>  <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit-learn" width="110"/>

#### A Python-based command-line tool that performs sentiment analysis on Amazon product reviews using state-of-the-art transformer models. It fetches real reviews, processes them, applies a sentiment model, and visualizes the results with helpfulness insights.
---
## 📦 Features

- Load real Amazon product reviews using the HuggingFace `datasets` library.
- Choose from 5 different pre-trained transformer models for sentiment analysis.
- Automatically maps star ratings to sentiments for evaluation.
- Generates classification reports and confusion matrices.
- Analyzes the impact of 'helpful vote' votes for each sentiment.
- Visualizes helpfulness scores across sentiment categories.
<br><br>

## 🔧 Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Packages:
- transformers
- datasets
- scikit-learn
- matplotlib
- seaborn
- numpy
- pandas
<br><br>

## 🚀 Usage
```
# Analyze 500 reviews, with DistilBERT model, in debug mode.
python main.py --num_reviews 500 --model_id 1 --verbose 2
```

### Arguments

| Argument      | Type | Default Value | Description                            |
|---------------|------|----------|---------------------------------------------|
| `--num_reviews` | *int*  | 1000 | Number of reviews to process (default: 100) |
| `--model_id`    | *int*  | 0 | Model to use (0–4, see below)                  |
| `--verbose`     | *int*  | 1 | Verbosity: 0=silent, 1=summary, 2=debug        |


## 🤖 Supported Models

| ID | Model Name |
|----|------------|
| 0 | [cardiffnlp/twitter-roberta-base-sentiment-latest (Twitter-roBERTa)](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) |
| 1 | [distilbert-base-uncased-finetuned-sst-2-english (DistilBERT)](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) |
| 2 | [finiteautomata/bertweet-base-sentiment-analysis (BERTweet)](https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis) |
| 3 | [nlptown/bert-base-multilingual-uncased-sentiment (Multilingual BERT)](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) |
| 4 | [siebert/sentiment-roberta-large-english (SiEBERT)](https://huggingface.co/siebert/sentiment-roberta-large-english) |


## 📊 Outputs
- Classification Report
- Confusion Matrix
- Helpfulness Score Distribution
- Saved Visualizations under ./visualizations/


## 📁 Project Structure
```
.
├── main.py
├── amazon_reviews_sentiment_analyzer.py
├── utils.py
├── requirements.txt
├── data/
│   └── raw_review_All_Beauty/... (Arrow dataset file)
└── visualizations/
    ├── Confusion_Matrix.png
    ├── Classification_Report.png
    └── Help_Distribution.png
```
---
## 📌 Notes
- The dataset file should be present under data/raw_review_All_Beauty/ as .arrow file format.
- Ensure model directories (if local) are correctly set for bertweet (model_id = 2).

## 🙌 Acknowledgments
- HuggingFace Transformers
- HuggingFace Datasets
- Scikit-learn
- Amazon Reviews Dataset
