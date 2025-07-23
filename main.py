"""
Main entry point for running sentiment analysis on Amazon product reviews.

This script:
- Parses CLI arguments for model selection and verbosity.
- Loads and analyzes Amazon reviews using a selected transformer model.
- Evaluates model predictions against review star ratings.
- Outputs metrics and visualizations.

Usage:
    python main.py --num_reviews 200 --model_id 1 --verbose 2
"""

from amazon_reviews_sentiment_analyzer import AmazonReviewsSentimentAnalyzer
import argparse
from utils import *
import logging

# Suppress verbose logs
tf_logging.set_verbosity_error()
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Argument parser setup
parser = argparse.ArgumentParser(
    description="Amazon Reviews Sentiment Analyzer CLI"
)

# Define arguments
parser.add_argument('--num_reviews', default=1000, type=non_negative_int,
                    help='Number of raw reviews to analyze (must be ≥ 0)')

parser.add_argument('--model_id', default=0, type=int,
                    choices=range(len(MODELS_NAMES)),
                    help='ID of the model to use for sentiment analysis')

parser.add_argument('--verbose', default=1, type=int, choices=[0, 1, 2],
                    help='Verbosity level: 0=silent, 1=summary, 2=debug')


def main() -> None:
    """
    Executes the main sentiment analysis workflow:
    - Parses command-line arguments.
    - Initializes logger and sentiment analyzer.
    - Loads and processes reviews.
    - Runs sentiment analysis and generates evaluation output.
    """
    # Parse the arguments from the command line
    args = parser.parse_args()
    setup_logger(args.verbose)

    analyzer = AmazonReviewsSentimentAnalyzer(args)
    analyzer.fetch_reviews()
    analyzer.batch_sentiment_analysis()
    analyzer.compare_sentiments()


if __name__ == "__main__":
    main()
