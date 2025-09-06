# Disable tokenizer parallelism warning (must be set before imports)
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
from typing import Any, Dict, List, Tuple, Union
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from transformers import pipeline, Pipeline
from tqdm import tqdm

from utils import *
from datasets import load_dataset


class AmazonReviewsSentimentAnalyzer:
    """
    Class for analyzing sentiment of Amazon product reviews.

    Provides methods to fetch, analyze, evaluate, and visualize review
    sentiment.
    """

    PREDICTED: str = "predicted_sentiment"
    HELPFUL: str = "helpful_vote"

    def __init__(self, args):
        """
        Initialize the sentiment analyzer.

        Args:
            args (Any): Parsed command-line arguments or config object
            containing:
                - num_reviews (int): Number of reviews to fetch from the
                    dataset.
                - verbose (int): Verbosity level:
                    0 = silent,
                    1 = summary,
                    2 = full debug output.
                - model_id (int): Index of the model to use for sentiment
                analysis (must correspond to index in MODELS_NAMES).

        Attributes:
            amazon_sentiment (List):
                Original rating-based sentiment labels.
            sentiment_analysis (List):
                Transformer-predicted sentiment results.
            sentiment_analyzer (Pipeline):
                Hugging Face sentiment analysis pipeline.
            reviews_df (pd.DataFrame):
                DataFrame containing the loaded and processed reviews.
            args (Any):
                The same config object passed in for global access.
        """

        # Reviews ranging from 1 to 5
        self.amazon_sentiment = []
        # Reviews based on Transformer model
        self.sentiment_analysis = []

        self.sentiment_analyzer = None
        self.reviews_df = None

        self.args = args

    def fetch_reviews(self, num_reviews: int = 100
                      ) -> List[Dict[str, Union[str, float, int, bool]]]:
        """
        Fetches and stores Amazon product reviews from the dataset.

        Args:
            num_reviews (int): Number of reviews to be fetched from Amazon
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
            # Set path based on OS
            path = self._get_dataset_path()
            # Load dataset
            dataset = self._load_dataset(path)
            # Get the specified number of reviews
            reviews = self._extract_reviews(dataset, num_reviews)
            # Extract only the required features
            filtered_reviews = self._filter_reviews(reviews)
            # Keep the filtered reviews
            self.reviews_df = pd.DataFrame(filtered_reviews)

            if self.args.verbose == 2:
                logging.debug(print_centered("Filtered Reviews"))
                print_formatted_dictionaries(filtered_reviews)

            # Keep a copy of only ID and sentiment for later comparison
            self.set_amazon_sentiment(filtered_reviews)
            return filtered_reviews

        except Exception as e:
            logging.error(f"Error fetching reviews: {e}")

    def set_amazon_sentiment(self, reviews: List[Dict[str, Any]]) -> None:
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

        # Only prints all details in 'debug' mode
        if self.args.verbose == 2:
            logging.debug(print_centered("Amazon rating"))
            print_formatted_dictionaries(self.amazon_sentiment)

    def init_sentiment_analyzer(self, model_id: int = 0):
        """
        Initialize the sentiment analyzer pipeline.

        Args:
            model_id (int): The model index to use. Defaults to 0.

        Raises:
            RuntimeError: If pipeline initialization fails.
        """
        try:
            logging.info("Initializing sentiment analyzer...")

            model_name = self._get_model_name(model_id)
            self.sentiment_analyzer = self._load_pipeline(model_name, model_id)

            logging.info("Sentiment analyzer initialized successfully")

        except Exception as e:
            logging.error(f"Error initializing sentiment analyzer: {e}")
            raise RuntimeError(
                "Failed to initialize sentiment analyzer.") from e

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment label and confidence score
        """
        if not self.sentiment_analyzer:
            # Sentiment analyzer model name
            self.init_sentiment_analyzer()

        try:
            if self.args.model_id == 2:
                # Truncate text if too long to handle token limit (bertweet)
                text = text[:128]
            else:
                text = text[:512]
            result = self.sentiment_analyzer(text)[0]

            mapped_label = LABEL_MAPPING.get(result['label'],
                                             result['label'].capitalize())

            return {
                'sentiment': mapped_label,
                'confidence': result['score']
            }

        except Exception as e:
            logging.error(f"Error analyzing sentiment: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.0}

    def analyze_helpfulness_by_sentiment(self) -> None:
        """
        Analyzes the helpfulness of reviews by sentiment category.

        Prints the mean, median, and count of helpful votes for each
        predicted sentiment group.
        """
        if self.reviews_df is None or self.reviews_df.empty:
            print("No data available.")
            return

        grouped = (self.reviews_df.groupby(self.PREDICTED)[self.HELPFUL].
                   agg(['mean', 'median', 'count']))

        grouped.index.name = None

        print(grouped.round(2).to_string())

    def batch_sentiment_analysis(self) -> pd.DataFrame:
        """
        Perform sentiment analysis on all reviews in the dataset.

        Returns:
            DataFrame with sentiment analysis results.
        """
        self.sentiment_analysis = []

        if self.reviews_df is None or self.reviews_df.empty:
            logging.error("No reviews to analyze. Please fetch reviews first.")
            return pd.DataFrame()

        logging.info("Performing sentiment analysis on all reviews...")
        texts = self._combine_review_texts()
        sentiments, confidences, results = self._analyze_batch(texts)

        self.sentiment_analysis = results

        if self.args.verbose == 2:
            logging.debug("\n--- Sentiment results ---")
            print_formatted_dictionaries(results)

        self.reviews_df[self.PREDICTED] = sentiments
        self.reviews_df['sentiment_confidence'] = confidences

        # Categorize helpfulness
        self.reviews_df['helpful_category'] = pd.cut(
            self.reviews_df[self.HELPFUL],
            bins=[-1, 0, 5, 20, float('inf')],
            labels=['None', 'Low', 'Medium', 'High']
        )

        logging.info("Sentiment analysis completed")
        return self.reviews_df

    def convert_rating(self) -> List[str]:
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

        # Converts the rating num 1-5 to sentiment
        for review in self.amazon_sentiment:
            rating_as_sentiment.append(RATING_TO_SENTIMENT[review['rating']])

        return rating_as_sentiment

    def extract_sentiment(self) -> List[str]:
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
        # Prints in case of summary or debug mode
        if self.args.verbose and self.args.verbose in [1, 2]:
            try:
                logging.info("Classification report")
                print(classification_report(y_true, y_pred,
                                            labels=labels, zero_division=0))

                logging.info("Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred, labels=labels)
                df_cm = pd.DataFrame(cm, index=labels, columns=labels)
                print(df_cm, "\n")

                logging.info("Helpfulness by Sentiment")
                self.analyze_helpfulness_by_sentiment()

            except ValueError as e:
                logging.error(f"Evaluation failed: {e}")

    def plot_confusion_matrix(self, y_true: List[str],
                              y_pred: List[str],
                              labels: List[str],
                              save_path:
                              str = './visualizations/Confusion_Matrix.png') \
            -> None:
        """
        Plot confusion matrix as heatmap.

        Args:
            y_true (List[str]): True sentiment labels.
            y_pred (List[str]): Predicted sentiment labels.
            labels (List[str]): List of all class labels.
            save_path (str): File path to save the figure.
        """
        model_name = MODEL_MAPPING[MODELS_NAMES[self.args.model_id]]
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        fig = plt.figure(figsize=(6, 5))
        plt.rcParams['font.family'] = 'DejaVu Sans'
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted Labels", color='crimson')
        plt.ylabel("True Labels", color='royalblue')
        plt.title(
            f"Sentiment Prediction vs True Labels\nModel: {model_name}",
            fontsize=9.2)
        plt.subplots_adjust(bottom=0.7)
        plt.tight_layout()

        self._save_fig(fig, save_path)

    def plot_classification(self, y_true: List[str],
                            y_pred: List[str],
                            save_path: str = './visualizations'
                                             '/Classification_Report.png') \
            -> None:
        """
        Plot classification report as heatmap.

        Args:
            y_true (List[str]): True sentiment labels.
            y_pred (List[str]): Predicted sentiment labels.
            save_path (str): File path to save the figure.
        """
        # report = classification_report(y_true, y_pred, output_dict=True)

        report = classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            output_dict=True,
            zero_division=0
        )

        df = pd.DataFrame(report).transpose()
        # Enforce row order
        df = df.loc[LABELS]

        fig = plt.figure(figsize=(6, 5))
        sns.heatmap(df.iloc[:, :-1], annot=True, cmap="YlGnBu")

        plt.title("Classification Report")
        plt.tight_layout()

        self._save_fig(fig, save_path)

    def plot_helpfulness_distribution(self,
                                      save_path: str
                                      = './visualizations/Help_Distribution'
                                        '.png') \
            -> None:
        """
        Plots a boxplot showing the distribution of helpful votes
        across predicted sentiment categories.
        """
        if self.reviews_df is None or self.reviews_df.empty:
            return

        fig = plt.figure(figsize=(6, 5))
        plt.rcParams['font.family'] = 'DejaVu Sans'

        sns.boxplot(
            data=self.reviews_df,
            x=self.PREDICTED,
            y=self.HELPFUL,
            hue=self.PREDICTED,
            palette='Accent',
            legend=False
        )

        plt.title("Helpfulness Votes by Sentiment", fontsize=12)
        plt.xlabel("Predicted Sentiment", color='crimson')
        plt.ylabel("Helpful Votes", color='royalblue')

        # Trims extreme outliers visually
        plt.ylim(0, 20)
        plt.tight_layout()

        self._save_fig(fig, save_path)

    def compare_sentiments(self) -> None:
        """
        Compare predicted sentiment with rating-based sentiment.

        Returns:
            None.
        """
        labels = LABELS

        y_true = self.convert_rating()
        y_pred = self.extract_sentiment()

        # Evaluates and prints the classification report and confusion matrix
        self.eval(y_true, y_pred)

        try:
            self.plot_confusion_matrix(y_pred, y_true, labels)
            self.plot_classification(y_pred, y_true)
            self.plot_helpfulness_distribution()

            plt.show()
        except ValueError as e:
            logging.error(f"Plotting failed: {e}")

    def _get_dataset_path(self) -> str:
        """
        Sets the path to the data files based on the user's operating system.

        Returns:
            str: Full path to the Amazon reviews dataset file,
                 adjusted for Windows or Unix-based systems.
        """
        try:
            # Set the path according to OS
            path = (Path.cwd() / 'data' / 'raw_review_All_Beauty' / '0.0.0' /
                    '16b76e' / 'amazon-reviews-2023-full.arrow')

        except Exception as e:
            logging.error(f"Error getting path: {e}")
            return None

        return str(path)

    # ----- Internal helper methods -----
    def _load_dataset(self, path: str) -> dict:
        """
        Loads the dataset from the specified path using the Hugging Face
        `load_dataset` function.

        Args:
            path (str): Full path to the Amazon reviews dataset file.

        Returns:
            dict: A dictionary containing the dataset split(s), typically with a
                  "train" key for access to the review data.
        """
        return load_dataset(
            path='arrow',
            data_files=path
        )

    def _extract_reviews(self, dataset: dict, num_reviews: int) \
            -> List[Dict[str, Any]]:
        """
        Extracts a specified number of reviews from the dataset.

        Args:
            dataset (dict): Dictionary containing the loaded dataset.
            num_reviews (int): Number of reviews to extract.

        Returns:
            list: A list of raw review entries.
        """
        logging.info("###########################")
        return [dataset["train"][i] for i in range(num_reviews)]

    def _filter_reviews(self, reviews: List[Dict[str, Any]]) \
            -> List[Dict[str, Any]]:
        """
        Filters raw review entries to extract only the required fields and
        assigns a unique ID to each review.

        Args:
            reviews (list): List of raw review entries.

        Returns:
            list: List of dictionaries containing filtered review data with keys:
                  'unique_id', 'user_id', 'title', 'text', 'rating',
                  'helpful_vote', 'verified_purchase'.
        """
        filtered_reviews = []
        for i, review in enumerate(reviews, start=1):
            filtered_reviews.append({
                'unique_id': i,
                'user_id': review.get('user_id', ''),
                'title': review.get('title', ''),
                'text': review.get('text', ''),
                'rating': review.get('rating', 0),
                'helpful_vote': review.get(self.HELPFUL, 0),
                'verified_purchase': review.get('verified_purchase', False)
            })
        return filtered_reviews

    def _combine_review_texts(self) -> List[str]:
        """
        Combines title and text from each review for analysis.

        Returns:
            List of concatenated review texts.
        """
        return (self.reviews_df['title'] + " " + self.reviews_df[
            'text']).tolist()

    def _analyze_batch(self, texts: List[str]) \
            -> Tuple[List[str], List[float], List[dict]]:
        """
        Runs sentiment analysis on a batch of review texts.

        Args:
            texts (List[str]): List of combined review texts.

        Returns:
            Tuple of (sentiments, confidences, full_results).
        """
        sentiments, confidences, results = [], [], []

        # Ensure the analyzer is initialized before tqdm
        if not self.sentiment_analyzer:
            self.init_sentiment_analyzer()

        for i, text in enumerate(tqdm(texts, desc="Analyzing Reviews",
                                      unit="review", leave=False)):
            result = self.analyze_sentiment(text)
            results.append({
                'unique_id': i + 1,
                'sentiment': LABEL_MAPPING.get(result['sentiment'],
                                               result[
                                                   'sentiment'].capitalize()),
                'confidence': result['confidence']
            })
            sentiments.append(result['sentiment'])
            confidences.append(result['confidence'])

        # Returns as a tuple
        return (sentiments, confidences, results)

    def _get_model_name(self, model_id: int) -> str:
        """
        Retrieve the model name from the global MODELS_NAMES list,
        optionally overridden by self.args.

        Args:
            model_id (int): Index of the model.

        Returns:
            str: The model name string.
        """
        if self.args:
            model_id = self.args.model_id
        return MODELS_NAMES[model_id]

    def _load_pipeline(self, model_name: str, model_id: int) -> Pipeline:
        """
        Load the sentiment analysis pipeline based on model name and ID.

        Args:
            model_name (str): The Hugging Face model name or path.
            model_id (int): The model ID index.

        Returns:
            Pipeline: The initialized Hugging Face pipeline.
        """
        tf_logging.set_verbosity_error()

        # load bertweet to reduce running time
        if model_id == 2:
            return pipeline(
                "sentiment-analysis",
                model="./models/bertweet",
                tokenizer="./models/bertweet",
                max_length=128,
                truncation=True,
                padding=True
            )

        return pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name
        )

    def _save_fig(self, fig: plt.Figure, save_path: str) -> None:
        """
        Save a matplotlib figure.

        Args:
            fig: The figure to save.
            save_path: Path where the figure will be saved.
        """
        # Create the path if does not exists and saves the result
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
