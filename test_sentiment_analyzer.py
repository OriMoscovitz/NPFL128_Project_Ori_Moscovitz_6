# run with: python -m unittest test_sentiment_analyzer.py

import unittest
from unittest.mock import patch, MagicMock
from amazon_reviews_sentiment_analyzer import AmazonReviewsSentimentAnalyzer
from utils import LABEL_MAPPING, non_negative_int
import pandas as pd


class TestAmazonReviewsSentimentAnalyzer(unittest.TestCase):

    def setUp(self):
        class Args:
            num_reviews = 5
            model_id = 0
            verbose = 0
        self.analyzer = AmazonReviewsSentimentAnalyzer(Args())

    def test_filter_reviews(self):
        raw = [
            {
                'user_id': 'u1',
                'title': 'T',
                'text': 'Good',
                'rating': 5.0,
                'helpful_vote': 3,
                'verified_purchase': True
            }
        ]
        result = self.analyzer._filter_reviews(raw)
        self.assertEqual(len(result), 1)
        self.assertIn('unique_id', result[0])
        self.assertEqual(result[0]['rating'], 5.0)

    def test_convert_rating(self):
        self.analyzer.amazon_sentiment = [
            {'rating': 1.0},
            {'rating': 3.0},
            {'rating': 5.0}
        ]
        expected = ['Negative', 'Neutral', 'Positive']
        self.assertEqual(self.analyzer.convert_rating(), expected)

    def test_extract_sentiment(self):
        self.analyzer.sentiment_analysis = [
            {'sentiment': 'Negative'},
            {'sentiment': 'Neutral'}
        ]
        self.assertEqual(self.analyzer.extract_sentiment(),
                         ['Negative', 'Neutral'])

    def test_label_mapping(self):
        self.assertEqual(LABEL_MAPPING['LABEL_2'], 'Positive')
        self.assertEqual(LABEL_MAPPING['NEG'], 'Negative')
        self.assertEqual(LABEL_MAPPING['NEUTRAL'], 'Neutral')

    def test_non_negative_int_valid(self):
        self.assertEqual(non_negative_int('10'), 10)

    def test_non_negative_int_invalid(self):
        with self.assertRaises(Exception):
            non_negative_int('-1')

    @patch('amazon_reviews_sentiment_analyzer.pipeline')
    def test_init_sentiment_analyzer(self, mock_pipeline):
        mock_pipeline.return_value = MagicMock()
        self.analyzer.init_sentiment_analyzer(0)
        self.assertIsNotNone(self.analyzer.sentiment_analyzer)

    @patch('amazon_reviews_sentiment_analyzer'
           '.AmazonReviewsSentimentAnalyzer.analyze_sentiment')
    def test_analyze_batch(self, mock_analyze):
        self.analyzer.reviews_df = pd.DataFrame({
            'title': ['A', 'B'],
            'text': ['Text1', 'Text2']
        })
        mock_analyze.side_effect = lambda x: {
            'sentiment': 'Positive', 'confidence': 0.9
        }
        sentiments, confidences, results = (
            self.analyzer._analyze_batch(['A Text1', 'B Text2'])
        )
        self.assertEqual(len(sentiments), 2)
        self.assertEqual(sentiments[0], 'Positive')
        self.assertTrue(all(r['confidence'] == 0.9 for r in results))


if __name__ == '__main__':
    unittest.main()
