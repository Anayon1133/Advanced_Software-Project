import pandas as pd
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk


nltk.download('vader_lexicon')


# Load the pre-trained sentiment analysis model
sentiment_analysis = pipeline("sentiment-analysis")
sia = SentimentIntensityAnalyzer()


# Function to perform sentiment analysis on a given text
def analyze_sentiment(text, max_length=512):
    # Split text into chunks of max_length tokens
    text_chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    
    # Analyze sentiment for each chunk
    sentiments = [sentiment_analysis(chunk)[0]['label'] for chunk in text_chunks]
    
    # Calculate the sentiment distribution for the chunks
    sentiment_distribution = {
        "POSITIVE": 0,
        "NEGATIVE": 0,
        "NEUTRAL": 0
    }
    for sentiment in sentiments:
        sentiment_distribution[sentiment] += 1
    
    # Return the most frequent sentiment
    return max(sentiment_distribution, key=sentiment_distribution.get)



def get_positive_negative_words(text):
    words = text.split()
    positive_words = [word for word in words if sia.polarity_scores(word)['compound'] > 0]
    negative_words = [word for word in words if sia.polarity_scores(word)['compound'] < 0]
    return positive_words, negative_words

# Apply the sentiment analysis function to the 'PreprocessedText' column
filename = "data/new_data.csv"  # Replace this with the path to your preprocessed CSV file
data = pd.read_csv(filename)
data['Sentiment'] = data['PreprocessedText'].apply(analyze_sentiment)
data['PositiveWords'], data['NegativeWords'] = zip(*data['PreprocessedText'].apply(get_positive_negative_words))

data = data.drop(columns=["PreprocessedText"])
output_filename = "file.csv"  # Replace this with the desired path for the output file
data.to_csv(output_filename, index=False)
# print(sentiment_counts)
