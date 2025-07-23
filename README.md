# Amazon Reviews Sentiment Analyzer 
<img src="https://upload.wikimedia.org/wikipedia/commons/4/41/Antu_amazon-mp3-store-source.svg" alt="Amazon Logo" width="70"/>  <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face" width="80"/><img src="https://upload.wikimedia.org/wikipedia/commons/6/67/Numpy-svgrepo-com.svg" alt="NumPy" width="70"/>  <img src="https://upload.wikimedia.org/wikipedia/commons/8/84/Matplotlib_icon.svg" alt="Matplotlib" width="70"/>  <img src="https://upload.wikimedia.org/wikipedia/commons/4/45/Logo-seaborn.png" alt="Seaborn" width="70"/>  <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit-learn" width="110"/>

### A Python-based command-line tool that performs sentiment analysis on Amazon product reviews using state-of-the-art transformer models. It fetches real reviews, processes them, applies a sentiment model, and visualizes the results with helpfulness insights.
---

## 📦 Features

- Load real Amazon product reviews using the HuggingFace `datasets` library.
- Choose from 5 different pre-trained transformer models for sentiment analysis.
- Automatically maps star ratings to sentiments for evaluation.
- Generates classification reports and confusion matrices.
- Analyzes the impact of 'helpful vote' votes for each sentiment.
- Visualizes helpfulness scores across sentiment categories.

---

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

## 🚀 Usage
```
# Analyze 500 reviews, with DistilBERT model, in debug mode.
python main.py --num_reviews 500 --model_id 1 --verbose 2
```

### Arguments

| Argument      | Type | Description                                |
|---------------|------|--------------------------------------------|
| `--num_reviews` | int  | Number of reviews to process (default: 100) |
| `--model_id`    | int  | Model to use (0–4, see below)               |
| `--verbose`     | int  | Verbosity: 0=silent, 1=summary, 2=debug     |

