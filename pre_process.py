from nltk.stem import WordNetLemmatizer, SnowballStemmer
import numpy as np
import nltk
from gensim.utils import simple_preprocess
import pandas as pd
from gensim.parsing.preprocessing import STOPWORDS
import re
import json
import os


def get_twitter_account():
    occur = {}
    df = pd.read_csv("dataset/covid19_tweets.csv")
    for text in df['text']:
        accounts = [t for t in text.split() if t.startswith('@')]
        if len(accounts) > 0:
            for account in accounts:
                if account in occur:
                    occur[account] += 1
                else:
                    occur[account] = 1
    occur = {k: v for k, v in occur.items() if v >= 10}
    # result = [{k: v} for k, v in sorted(occur.items(), key=lambda item: item[1], reverse=True)]
    with open('dataset/twitter_accounts.txt', 'w') as outfile:
        json.dump(occur, outfile)


np.random.seed(2018)
nltk.download("wordnet")
stemmer = SnowballStemmer('english')
if not os.path.exists("dataset/twitter_accounts.txt"):
    get_twitter_account()
with open("dataset/twitter_accounts.txt") as json_file:
    twitter_accounts = json.load(json_file)


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text):
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    text = re.sub(regex, '#href_link', text)
    accounts = [t for t in text.split() if t.startswith('@')]
    for acc in accounts:
        if acc not in twitter_accounts:
            text = text.replace(acc, "twitter_account")

    result = []
    for token in simple_preprocess(text):
        if token not in STOPWORDS:
            result.append(lemmatize_stemming(token))
    return result


def main():
    documents = pd.read_csv("./dataset/covid19_tweets.csv")
    print("len doc = ", len(documents))
    processed_docs = documents['text'].map(preprocess)
    print("len processed_docs = ", len(processed_docs))
    processed_text = {"text_processed": list()}
    for doc in processed_docs:
        processed_text['text_processed'].append(" ".join(doc))
    df = pd.DataFrame(processed_text)
    df.to_csv("./dataset/covid19_tweets_processed.csv")
    processed_text['text'] = documents['text']
    df = pd.DataFrame(processed_text)
    df.to_csv("./dataset/covid19_tweets_processed_full.csv")


if __name__ == '__main__':
    main()
