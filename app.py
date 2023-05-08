from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from collections import Counter

app = Flask(__name__, template_folder='templates')
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///sentiment_data.db"
db = SQLAlchemy(app)

class CountrySentiment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    country = db.Column(db.String(50), unique=True, nullable=False)
    positive_words = db.Column(db.String(1000), nullable=False)
    negative_words = db.Column(db.String(1000), nullable=False)
    sentiment_score = db.Column(db.String(1000), nullable=False)

def get_top_words(words):
    # Convert the comma-separated string of words into a list
    words_list = words.split(',')
    # Count the frequency of each word
    word_counts = Counter(words_list)
    # Get the top 20 most common words
    top_words = word_counts.most_common(20)
    # Return a list of just the words (not the counts)
    return [word[0] for word in top_words]

def generate_wordclouds():
    countries = CountrySentiment.query.all()
    img_folder = "static/wordclouds"

    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    for country_data in countries:
        positive_words = country_data.positive_words.split(',')
        negative_words = country_data.negative_words.split(',')

        words = ' '.join(positive_words + negative_words)
        wordcloud = WordCloud(width=800, height=800, background_color='black', min_font_size=10).generate(words)

        img_filename = f"{country_data.country}_wordcloud.png"
        img_path = os.path.join(img_folder, img_filename)
        wordcloud.to_file(img_path)


with app.app_context():
    db.create_all()

    # Load data into the database
    data_file = "data/sentiment_data.csv"
    data = pd.read_csv(data_file)
    if CountrySentiment.query.count() == 0:
        for index, row in data.iterrows():
            sentiment = CountrySentiment(
                country=row["Country"],
                positive_words=row["PositiveWords"],
                negative_words=row["NegativeWords"],
                sentiment_score=row["Sentiment"],
            )
            db.session.add(sentiment)
        db.session.commit()

    # Generate and save all wordclouds
    # generate_wordclouds()

@app.route("/")
def index():
    countries = db.session.query(CountrySentiment.country).all()
    country_list = [country[0] for country in countries]
    return render_template("index.html", countries=country_list)



@app.route("/country-sentiment", methods=["POST"])
def country_sentiment():
    country = request.form["country"]
    sentiment_data = CountrySentiment.query.filter_by(country=country).first()
    top_positive_words = get_top_words(sentiment_data.positive_words)
    top_negative_words = get_top_words(sentiment_data.negative_words)

    img_folder = "static/wordclouds"
    img_filename = f"{country}_wordcloud.png"
    img_path = os.path.join(img_folder, img_filename)

    return render_template("country_sentiment.html", country=country, sentiment_data=sentiment_data, wordcloud_path=img_path, top_positive_words=top_positive_words, top_negative_words=top_negative_words)


if __name__ == "__main__":
    app.run(debug=True)
