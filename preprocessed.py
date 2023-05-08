import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import string

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')



def preprocess_text(text):
    # Lowercase
    text = text.lower()

    # Remove special characters, numbers, and punctuation
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize words
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Return preprocessed text
    return ' '.join(words)



# Read the CSV file
filename = "data/countries.csv" # Replace this with the path to your CSV file
data = pd.read_csv(filename)
col_data = ['DateTime',	'URL', 'SharingImage', 'LangCode', 'DocTone','DomainCountryCode', 'Location','Lat',	'Lon',	'CountryCode',	'Adm1Code',	'GeoType','the_geom',	'Adm2Code','Title']
data = data.drop(columns=col_data)
# Apply the preprocessing function to the 'ContextualText' column
data['PreprocessedText'] = data['ContextualText'].apply(preprocess_text)
# Drop ContextualText column
data = data.drop(columns=['ContextualText'])

data = data.drop_duplicates(subset=['PreprocessedText'])
merged_news = data.groupby('Country')['PreprocessedText'].apply(' '.join).reset_index()
# Save the preprocessed data to a new CSV file
preprocessed_csv_filename = "file.csv"
merged_news.to_csv(preprocessed_csv_filename, index=False)

print(merged_news.head())

