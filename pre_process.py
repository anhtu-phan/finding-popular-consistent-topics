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


def preprocess_url(text, type_pre='REPLACE'):
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    if type_pre == 'REPLACE':
        text = re.sub(regex, '#href_link', text)
    else:
        text = re.sub(regex, '', text)
    return text


def preprocess_twitter_account(text, type_pre='REPLACE'):
    texts = text.split()
    accounts = [(i, t) for i, t in enumerate(texts) if t.startswith('@')]
    for index, acc in accounts:
        if type_pre == 'REPLACE':
            if acc not in twitter_accounts:
                texts[index] = "twitter_account"
        else:
            texts[index] = ""
    return " ".join(texts)


def preprocess(text, types_pre: list() = ['REMOVE', 'REMOVE']):
    if types_pre[0] is not None:
        text = preprocess_url(text, type_pre=types_pre[0])
    if types_pre[1] is not None:
        text = preprocess_twitter_account(text, type_pre=types_pre[1])

    result = []
    for token in simple_preprocess(text):
        if token not in STOPWORDS:
            result.append(lemmatize_stemming(token))
    return " ".join(result)


def process():
    documents = pd.read_csv("./dataset/covid19_tweets.csv")
    print("len doc = ", len(documents))
    remove_url = documents['text'].apply(preprocess, args=(['REMOVE', None],))
    remove_twitter_account = documents['text'].apply(preprocess, args=(['REMOVE', 'REMOVE'],))
    remove_url_replace_twitter_account = documents['text'].apply(preprocess, args=(['REMOVE', 'REPLACE'],))
    remove_twitter_account_replace_url = documents['text'].apply(preprocess, args=(['REPLACE', 'REMOVE'],))
    replace_twitter_account_and_url = documents['text'].apply(preprocess, args=(['REPLACE', 'REPLACE'],))
    processed_text = {"date": documents["date"], "remove_url": remove_url, "remove_twitter_account": remove_twitter_account,
                      "remove_url_replace_twitter_account": remove_url_replace_twitter_account,
                      "remove_twitter_account_replace_url": remove_twitter_account_replace_url,
                      "replace_twitter_account_and_url": replace_twitter_account_and_url}

    df = pd.DataFrame(processed_text)
    print("len processed_text = ", len(df))
    df.to_csv("./dataset/covid19_tweets_processed.csv")


if __name__ == '__main__':
    process()
