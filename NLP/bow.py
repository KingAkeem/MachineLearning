import pandas as pd
import re
import warnings

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from bs4 import BeautifulSoup

warnings.filterwarnings('ignore', category=UserWarning, module='bs4')

train = pd.read_csv("../datasets/labeledTrainData.tsv", header=0,
                    delimiter="\t", quoting=3)


def clean_html(html_docs):

    if type(html_docs) != str:
        clean_docs = list()
        for i, html in enumerate(html_docs):
            if i % 2 == 0:
                print("Preprocessing Page {i} of {t}".format(i=i, t=len(html_docs)))
            # Removing all html tags
            text = BeautifulSoup(html).get_text()

            # Removing all nonalphanumeric characters
            letters_only = re.sub("[^a-zA-Z]", " ", text)

            # Turning document into list of words
            words = letters_only.lower().split()

            # Appending cleaned document to list of cleaned documents
            clean_docs.append(" ".join(words))
    else:
        # Removing all html tags
        text = BeautifulSoup(html_docs).get_text()

        # Removing all nonalphanumeric characters
        letters_only = re.sub("[^a-zA-Z]", " ", text)

        # Turning document into lower case words
        words = letters_only.lower()

    return (words,)


clean_docs = clean_html(train['review'])

vec = CountVectorizer(analyzer='word', stop_words='english', max_features=5000)
features = vec.fit_transform(train['review']).toarray()

forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(features, train["sentiment"])

clean_test = clean_html(train['review'][3])

test_features = vec.transform(clean_test)
result = forest.predict(test_features)
