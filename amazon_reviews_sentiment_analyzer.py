from datasets import load_dataset
from typing import Any, Dict, List
from utils import print_formatted_dictionaries
import pandas as pd
from transformers import pipeline
from utils import RATING_TO_SENTIMENT, LABELS
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def convert_rating_to_sentiment(rating: float) -> str:
    """
    Convert numerical rating to sentiment category.

    Args:
        rating: Numerical rating (1-5)

    Returns:
        Sentiment category (positive, neutral, negative)
    """
    if rating <= 2:
        return 'negative'
    elif rating == 3:
        return 'neutral'
    else:  # rating >= 4
        return 'positive'


class AmazonReviewsSentimentAnalyzer:
    """
    A class for analyzing sentiment of Amazon product reviews.

    This classd maintains a collection of reviews and provides methods
    to analyze sentiment, and filter reviews.
    """

    def __init__(self, args):
        """
        Initialize the sentiment analyzer.

        Args:
            None
        """
        # Original reviews full data
        self.reviews = []
        # Reviews ranging from 1 to 5
        self.amazon_sentiment = []
        # Reviews based on Transformer model
        self.sentiment_analysis = []

        self.sentiment_analyzer = None
        self.reviews_df = None

        self.args = args

    def fetch_reviews(self, num_reviews: int = 100) -> List:
        """
        Initialize the sentiment analyzer.

        Args: num_reviews (int): Number of reviews to be fetched from Amazon
        DB repository. If not provided, defaults to num_reviews = 100.

        Returns:
            List with the selected amount of reviews.
            Saving only the attributes:
                # user_id
                # title
                # text
                # rating
                # helpful_vote
                # verified_purchase
        """
        if self.args:
            num_reviews = self.args.num_reviews

        try:
            # Load dataset
            dataset = load_dataset(
                path='arrow',
                data_files=r'O:\Charles\NPFL128\NPFL128_project\data'
                           r'\raw_review_All_Beauty\0.0.0\16b76e0823d73bb'
                           r'8cff1e9c5e3e37dbc46ae3daee380417ae141f5e67d'
                           r'3ea8e8\amazon-reviews-2023-full.arrow'
            )

            reviews = []

            # Get the specified number of reviews
            for i in range(num_reviews):
                reviews.append(dataset["train"][i])

            unique_id = 1

            # Extract only the required features
            filtered_reviews = []
            for review in reviews:
                filtered_reviews.append({
                    'unique_id': unique_id,
                    'user_id': review.get('user_id', ''),
                    'title': review.get('title', ''),
                    'text': review.get('text', ''),
                    'rating': review.get('rating', 0),
                    'helpful_vote': review.get('helpful_vote', 0),
                    'verified_purchase': review.get('verified_purchase', False)
                })
                unique_id += 1

            self.reviews_df = pd.DataFrame(filtered_reviews)

            if self.args.verbose:
                print("--- filtered_reviews ---")
                print_formatted_dictionaries(filtered_reviews)
            self.set_amazon_sentiment(filtered_reviews)

        except Exception as e:
            print(f"Error fetching reviews: {e}")

    def set_amazon_sentiment(self, reviews) -> None:
        """
        Sets the self.amazon_sentiment with the reviews id and their
        corresponding rating

        Args:
            reviews (list): List with the filtered reviews

        Returns:
            None
        """

        # Clear existing data
        self.amazon_sentiment = []

        for review in reviews:
            self.amazon_sentiment.append({
                'unique_id': review.get('unique_id', ''),
                'rating': review.get('rating', 0),
            })

        if self.args.verbose:
            print("--- Amazon Sentiment ---")
            print_formatted_dictionaries(self.amazon_sentiment)

    def init_sentiment_analyzer(self, model_name: str = None):
        """
        Initialize the sentiment analysis model.

        Args:
            model_name: Hugging Face model for sentiment analysis
        """
        try:
            print("Initializing sentiment analyzer...")
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name
            )
            print("Sentiment analyzer initialized successfully")

        except Exception as e:
            print(f"Error initializing sentiment analyzer: {e}")
            # Fallback to default model
            self.sentiment_analyzer = pipeline("sentiment-analysis")

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment label and confidence score
        """
        if not self.sentiment_analyzer:
            s = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            self.init_sentiment_analyzer(s)

        try:
            # Truncate text if too long to handle token limit
            text = text[:512]
            result = self.sentiment_analyzer(text)[0]

            # Map different model outputs to standard labels
            label_mapping = {
                'POSITIVE': 'positive',
                'NEGATIVE': 'negative',
                'NEUTRAL': 'neutral',
                'LABEL_0': 'negative',
                'LABEL_1': 'neutral',
                'LABEL_2': 'positive'
            }

            mapped_label = label_mapping.get(result['label'],
                                             result['label'].lower())

            return {
                'sentiment': mapped_label,
                'confidence': result['score']
            }

        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.0}

    def batch_sentiment_analysis(self) -> pd.DataFrame:
        """
        Perform sentiment analysis on all reviews in the dataset.

        Returns:
            DataFrame with sentiment analysis results
        """
        # Clear existing data
        self.sentiment_analysis = []

        if self.reviews_df is None or self.reviews_df.empty:
            print("No reviews to analyze. Please fetch reviews first.")
            return pd.DataFrame()

        print("Performing sentiment analysis on all reviews...")

        # Combine title and text for more comprehensive analysis
        combined_text = self.reviews_df['title'] + " " + self.reviews_df[
            'text']

        # Clear existing data
        sentiments = []
        confidences = []

        for i, text in enumerate(combined_text):
            if i % 100 == 0:
                print(f"Processed {i}/{len(combined_text)} reviews")

            result = self.analyze_sentiment(text)

            self.sentiment_analysis.append({
                'unique_id': i + 1,
                'sentiment': result['sentiment'],
                'confidence': result['confidence'],
            })

            sentiments.append(result['sentiment'])
            confidences.append(result['confidence'])

        if self.args.verbose:
            print("--- sentiment_results ---")
            print_formatted_dictionaries(self.sentiment_analysis)

        # Add sentiment results to dataframe
        self.reviews_df['predicted_sentiment'] = sentiments
        self.reviews_df['sentiment_confidence'] = confidences

        print("Sentiment analysis completed")
        return self.reviews_df

    def convert_rating(self) -> List:
        """
        Converts the original rating values from 1.0 to 5.0 into sentiment.
        < 3.0 => Negative
          3.0 => Neutral
        > 3.0 => Positive

        Args:
            None

        Returns:
            List of strings (positive / neutral / negative)
        """
        rating_as_sentiment = []

        for review in self.amazon_sentiment:
            rating_as_sentiment.append(RATING_TO_SENTIMENT[review['rating']])

        return rating_as_sentiment

    def extract_sentiment(self) -> List:
        """
        Extractcs only sentiment from the sentiment returned.
        Ignores the unique_id and confidence attributes to be able
        to feed it later into sklearn.metrics:
            classification_report, confusion_matrix

        Args:
            None

        Returns:
            List of strings (positive / neutral / negative)
        """
        extracted_sentiments = []

        for sentiment in self.sentiment_analysis:
            extracted_sentiments.append(sentiment['sentiment'])

        return extracted_sentiments

    def eval(self, y_pred: list = None, y_true: list = None):
        """
        Evaluates the model sentiment analysis by comparing
        the analyzed sentiment to the original rating.


        Args:
            y_pred (list): Predicted labels.
            y_true (list): True labels.


        Returns:
            None. Prints classification report and confusion matrix.
        """
        labels = LABELS
        print(classification_report(y_true, y_pred, labels=labels))
        print(confusion_matrix(y_true, y_pred, labels=labels))

    def plot_classification_results(self,
            y_true: List[str],
            y_pred: List[str],
    ) -> None:
        """
        Visualize confusion matrix and classification report.

        Args:
            y_true (List[str]): Ground truth sentiment labels.
            y_pred (List[str]): Predicted sentiment labels.
            labels (List[str]): List of class labels to include in the matrix.
        """
        labels = LABELS
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

    def compare_sentiments(self) -> Dict[str, float]:
        """
        Compare predicted sentiment with rating-based sentiment.

        Returns:
            Dictionary with comparison metrics
        """

        y_true = self.convert_rating()
        y_pred = self.extract_sentiment()

        self.eval(y_pred, y_true)
        self.plot_classification_results(y_pred, y_true)
