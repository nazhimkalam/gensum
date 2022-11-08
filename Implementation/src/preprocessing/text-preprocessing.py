import pandas as pd

# Dataset
newsSummaryPath = "../../dataset/news_summary/news_summary.csv"
moviePath = {
    "small": "../../dataset/movie/movie_reviews_small.csv",
    "medium": "../../dataset/movie/movie_reviews_medium.csv"
}
gigawordPath = {
    "medium": "../../dataset/generalization/gigaword_medium.csv",
    "large": "../../dataset/generalization/gigaword_large.csv",
    "xlarge": "../../dataset/generalization/gigaword_xlarge.csv",
    "xxlarge": "../../dataset/generalization/gigaword_xxlarge.csv"
}

# performing text pre processing steps on the dataset for abstractive text summarization only
# 1. Removing punctuations
# 2. Removing special characters
# 3. Removing extra spaces
# 4. Removing stopwords
# 5. Converting to lowercase

newsSummaryDataset = pd.read_csv(newsSummaryPath, encoding='latin-1')
movieSmallDataset = pd.read_csv(moviePath["small"], encoding='latin-1')
movieMediumDataset = pd.read_csv(moviePath["medium"], encoding='latin-1')
gigawordMediumDataset = pd.read_csv(gigawordPath["medium"], encoding='latin-1')
gigawordLargeDataset = pd.read_csv(gigawordPath["large"], encoding='latin-1')
gigawordXLargeDataset = pd.read_csv(gigawordPath["xlarge"], encoding='latin-1')
gigawordXXLargeDataset = pd.read_csv(gigawordPath["xxlarge"], encoding='latin-1')

# Text Preprocessing News Summary Dataset
# Drop the columns that are not required expect the text and summary columns
newsSummaryDataset = newsSummaryDataset.drop(['author', 'date', 'headlines', 'read_more'], axis=1)
newsSummaryDataset

# Renaming columns
newsSummaryDataset = newsSummaryDataset.rename(columns={'text': 'summary', 'ctext': 'text'})
newsSummaryDataset.shape

# Drop NA values
newsSummaryDataset = newsSummaryDataset.dropna()
newsSummaryDataset.shape

# Drop duplicates values
newsSummaryDataset = newsSummaryDataset.drop_duplicates("text")
newsSummaryDataset.shape

# Performing Contraction Mapping [Expansion] eg:- "aren't" ==> "are not"
import contractions
newsSummaryDataset['text'] = newsSummaryDataset['text'].apply(lambda x: [contractions.fix(word) for word in x.split()])
newsSummaryDataset['text'] = newsSummaryDataset['text'].apply(lambda x: ' '.join(x))

newsSummaryDataset['summary'] = newsSummaryDataset['summary'].apply(lambda x: [contractions.fix(word) for word in x.split()])
newsSummaryDataset['summary'] = newsSummaryDataset['summary'].apply(lambda x: ' '.join(x))


# We will perform the below preprocessing tasks for our data:
# 1. Covert everything to lowercase
# 2. Remove HTML tags
# 3. Contraction mapping
# 4. Remove (‘s)
# 5. Remove any text inside the parenthesis ( )
# 6. Eliminate punctuations and special characters
# 7. Remove stopwords
# 8. Remove short words

# 1. Covert everything to lowercase
newsSummaryDataset['text'] = newsSummaryDataset['text'].apply(lambda x: x.lower())
newsSummaryDataset['summary'] = newsSummaryDataset['summary'].apply(lambda x: x.lower())

# 2. Remove HTML tags
from bs4 import BeautifulSoup
newsSummaryDataset['text'] = newsSummaryDataset['text'].apply(lambda x: BeautifulSoup(x, "html.parser").text)
newsSummaryDataset['summary'] = newsSummaryDataset['summary'].apply(lambda x: BeautifulSoup(x, "html.parser").text)

# 3. Contraction mapping
# Contraction Mapping [Expansion] eg:- "aren't" ==> "are not"
import contractions

newsSummaryDataset['text'] = newsSummaryDataset['text'].apply(lambda x: [contractions.fix(word) for word in x.split()])
newsSummaryDataset['text'] = newsSummaryDataset['text'].apply(lambda x: ' '.join(x))

newsSummaryDataset['summary'] = newsSummaryDataset['summary'].apply(lambda x: [contractions.fix(word) for word in x.split()])
newsSummaryDataset['summary'] = newsSummaryDataset['summary'].apply(lambda x: ' '.join(x))

# 4. Remove (‘s)
import re
def remove_s(text):
    text = re.sub("'s", "", text)
    return text

newsSummaryDataset['text'] = newsSummaryDataset['text'].apply(lambda x: remove_s(x))
newsSummaryDataset['summary'] = newsSummaryDataset['summary'].apply(lambda x: remove_s(x))

# 5. Remove any text inside any form of parenthesis ( ) [] {} < >
def remove_content_between_parenthsis(text):
    return re.sub(r'\([^)]*\)', '', text)

newsSummaryDataset['text'] = newsSummaryDataset['text'].apply(lambda x: remove_content_between_parenthsis(x))
newsSummaryDataset['summary'] = newsSummaryDataset['summary'].apply(lambda x: remove_content_between_parenthsis(x))

# 6. Eliminate punctuations and special characters
import string
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

newsSummaryDataset['text'] = newsSummaryDataset['text'].apply(lambda x: remove_punctuation(x))
newsSummaryDataset['summary'] = newsSummaryDataset['summary'].apply(lambda x: remove_punctuation(x))

# 7. Remove stopwords
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

newsSummaryDataset['text'] = newsSummaryDataset['text'].apply(lambda x: remove_stopwords(x))
newsSummaryDataset['summary'] = newsSummaryDataset['summary'].apply(lambda x: remove_stopwords(x))

# 8. Remove short words
def remove_shortwords(text):
    return ' '.join([word for word in text.split() if len(word) > 2])

newsSummaryDataset['text'] = newsSummaryDataset['text'].apply(lambda x: remove_shortwords(x))
newsSummaryDataset['summary'] = newsSummaryDataset['summary'].apply(lambda x: remove_shortwords(x))


# displaying the first row text and summary
newsSummaryDataset.iloc[0]

# Remove the rows that have empty text or summary
def remove_empty_rows(text, summary):
    return (text != '') & (summary != '')

newsSummaryDataset = newsSummaryDataset[newsSummaryDataset.apply(lambda x: remove_empty_rows(x['text'], x['summary']), axis=1)]
newsSummaryDataset.shape

# remove extra lines and trim spaces
def remove_extra_lines(text):
    return text.strip()

newsSummaryDataset['text'] = newsSummaryDataset['text'].apply(lambda x: remove_extra_lines(x))
newsSummaryDataset['summary'] = newsSummaryDataset['summary'].apply(lambda x: remove_extra_lines(x))

# Removing Emojis from the text
import re
def remove_emojis(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

newsSummaryDataset['text'] = newsSummaryDataset['text'].apply(lambda x: remove_emojis(x))
newsSummaryDataset['summary'] = newsSummaryDataset['summary'].apply(lambda x: remove_emojis(x))

# Removing URLs
import re
def remove_urls(text):
    return re.sub(r'http\S+', '', text)

newsSummaryDataset['text'] = newsSummaryDataset['text'].apply(lambda x: remove_urls(x))
newsSummaryDataset['summary'] = newsSummaryDataset['summary'].apply(lambda x: remove_urls(x))

# Saving the cleaned data to a csv file
newsSummaryDataset.to_csv('cleaned_news_summary.csv', index=False)

