# Standard library imports
import argparse
import logging
from typing import Any, Dict, List, Optional

# Third-party library imports
from transformers import logging as tf_logging


# Models names that could be used
MODELS_NAMES = [
    'cardiffnlp/twitter-roberta-base-sentiment-latest',
    'distilbert-base-uncased-finetuned-sst-2-english',
    'finiteautomata/bertweet-base-sentiment-analysis',
    'nlptown/bert-base-multilingual-uncased-sentiment',
    'siebert/sentiment-roberta-large-english',
]

# Model name mapping for figure plotting
MODEL_MAPPING = {
    'cardiffnlp/twitter-roberta-base-sentiment-latest': 'Twitter-roBERTa',
    'distilbert-base-uncased-finetuned-sst-2-english': 'DistilBERT',
    'finiteautomata/bertweet-base-sentiment-analysis': 'BERTweet',
    'nlptown/bert-base-multilingual-uncased-sentiment': 'BERT Multilingual',
    'siebert/sentiment-roberta-large-english': 'SiEBERT',
}

# Map different model outputs to standard labels
LABEL_MAPPING = {
    'POSITIVE': 'Positive',
    'NEGATIVE': 'Negative',
    'NEUTRAL': 'Neutral',
    'LABEL_0': 'Negative',
    'LABEL_1': 'Neutral',
    'LABEL_2': 'Positive',
    'POS': 'Positive',
    'NEG': 'Negative',
    'NEU': 'Neutral',
}

# Maps numeric star ratings to sentiment categories for evaluation purposes
RATING_TO_SENTIMENT = {
    1.0: 'Negative',
    2.0: 'Negative',
    3.0: 'Neutral',
    4.0: 'Positive',
    5.0: 'Positive',
}

# Ordered list of sentiment labels used in classification reports and plots
LABELS = [
    'Positive',
    'Neutral',
    'Negative',
]

# Console Color Formatting
RESET = '\033[0m'
BLUE = '\033[94m'
GREEN = '\033[92m'
RED = '\033[91m'


# Logger Setup
def setup_logger(level: int = 1) -> None:
    """
    Sets up a logger with colored output based on the logging level.

    Args:
        level (int):
        The logging level:
            0 = WARNING,
            1 = INFO,
            2 = DEBUG.
        Defaults to 1 (INFO).

    Returns:
        None
    """
    level_map = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

    class ColorFormatter(logging.Formatter):
        """
        Custom formatter that adds color to log messages
        based on their level.
        """
        LEVEL_COLORS = {
            logging.DEBUG: BLUE,
            logging.INFO: GREEN,
            logging.ERROR: RED,
        }

        def format(self, record: logging.LogRecord) -> str:
            """
            Overrides the format method to add color to the log
            message based on its level.

            Args:
                record (logging.LogRecord): The log record to format.

            Returns:
                str: The formatted log message with color.
            """
            color = self.LEVEL_COLORS.get(record.levelno, RESET)
            message = super().format(record)
            return f"{color}{message}{RESET}"

    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter("[%(levelname)s] %(message)s"))
    logging.root.handlers = [handler]
    logging.root.setLevel(level_map.get(level, logging.INFO))


# Validation for non-negative number of reviews argument from user
def non_negative_int(value: str) -> int:
    """
    Validates that the input string represents a non-negative integer.

    Args:
        value (str): The input string to validate.

    Returns:
        int: The validated non-negative integer.

    Raises:
        argparse.ArgumentTypeError: If the value is negative or not an integer.
    """
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(
            "--num_reviews must be a non-negative integer")
    return ivalue


def print_formatted_dictionaries(
    list_of_dicts: List[Dict[Any, Any]],
    truncate_keys: Optional[List[str]] = None,
    max_length: int = 120
) -> None:
    """
    Prints a list of dictionaries in a readable format.

    Long string values for specified keys (e.g., 'text', 'title') are truncated
    for display purposes.

    Args:
        list_of_dicts (List[Dict[Any, Any]]):
            The list of dictionaries to print.
        truncate_keys (Optional[List[str]]):
            Keys whose values should be truncated
            if they are long strings. Defaults to ['text', 'title'].
        max_length (int):
            Maximum string length before truncation. Defaults to 120.
    """
    if truncate_keys is None:
        truncate_keys = ['text', 'title']

    if not isinstance(list_of_dicts, list):
        print("Error: Input must be a list of dictionaries.")
        return

    for index, dictionary in enumerate(list_of_dicts, 1):
        print(f"\n--- Entry {index} ---")
        for key, value in dictionary.items():
            if key in truncate_keys and isinstance(value, str) and len(
                    value) > max_length:
                value = value[:max_length] + '...'
            print(f"{key:<20}: {value}")
        print("-" * 40)


def print_centered(text: str) -> None:
    """
    Prints the given text centered within a 50-character wide line.

    The text is padded with '=' characters on both sides,
    with a single space on each side of the text.

    Args:
        text (str): The text to center and print.
    """

    total_length = 50
    padding = total_length - len(text) - 2  # 2 for the '=' on both sides
    left = padding // 2
    right = padding - left
    print("=" * left + " " + text + " " + "=" * right)
