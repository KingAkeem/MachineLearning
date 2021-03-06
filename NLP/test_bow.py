import pandas as pd
import warnings
import re
import requests

from bs4 import BeautifulSoup
from requests.exceptions import HTTPError, ConnectionError
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sys import exit

warnings.filterwarnings('ignore', category=UserWarning, module='bs4')


def clean_html(html_docs):

    if type(html_docs) != str:
        clean_docs = list()
        for i, html in enumerate(html_docs):
            if i % 2 == 0:
                print("Preprocessing Page {i} of {t}".format(i=i+1,
                                                             t=len(html_docs)))
            # Removing all nonalphanumeric characters
            letters_only = re.sub("[^a-zA-Z]", " ", str(html))
            # Turning document into list of words
            letters_only = letters_only.replace(",", " ")
            words = letters_only.lower().split()
            # Appending cleaned document to list of cleaned documents
            clean_docs.append(" ".join(words))

        return clean_docs
    else:
        # Removing all nonalphanumeric characters
        letters_only = re.sub("[^a-zA-Z]", " ", html_docs)
        # Turning document into lower case words
        words = letters_only.lower()
        return words


# Getting user inputted url
while True:
    try:
        url = input('enter url (enter q to quit): ')
        if url == 'q':
            exit(0)
        resp = requests.get(url)
        break
    except (HTTPError, ConnectionError):
        print('Try again.\n')

# Cleaning html text
text = BeautifulSoup(resp.text).get_text()
clean_text = clean_html(text)

# Reading in data and splitting into training and testing set
data = pd.read_csv('../datasets/training_data.csv', header=0, delimiter=',')
train, test = train_test_split(data)

# Creating Count vectorizer and training it
vec = CountVectorizer(analyzer='word', stop_words='english', max_features=5000)
word_features = vec.fit_transform(train['content'].values.astype('U'))
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(word_features, train['class'])
test_features = vec.transform((clean_text,))
result = forest.predict(test_features)
if result == 1:
    print('This is a sports website.')
if result == 2:
    print('This is a news webiste.')
