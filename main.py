from amazon_reviews_sentiment_analyzer import AmazonReviewsSentimentAnalyzer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_reviews', default=300, type=int,
                    help='number of raw reviews to fetch from the db ('
                         'default: %(default)s)')
parser.add_argument('--verbose', default=False, type=int,
                    help='Controls printing')


if __name__ == "__main__":
    # parse the arguments from the command line
    args = parser.parse_args()

    analyzer = AmazonReviewsSentimentAnalyzer(args)
    analyzer.fetch_reviews()
    analyzer.batch_sentiment_analysis()
    analyzer.compare_sentiments()


