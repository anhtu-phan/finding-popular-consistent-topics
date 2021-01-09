import pandas as pd
import re
import os
from numpy import asarray, save, load
import time
from datetime import datetime


def get_stop_words():
    stop_words = []
    with open("dataset/stopwords.txt", 'r') as f:
        for line in f:
            stop_words.append(line.rstrip())
    return stop_words


def pre_process(path):
    data = pd.read_csv(path)
    data_processed = {'text_processed': list()}
    stop_words = get_stop_words()
    for _, row in data.iterrows():
        # data_processed['date'].append(row['date'])
        text = row['text']
        # data_processed['text'].append(text)
        text_processed = text.lower()
        text_processed = text_processed.replace("\n", " ")
        regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
        text_processed = re.sub(regex, '#href_link', text_processed)
        text_processed = re.sub("[^a-z0-9@#:/_ ]", "", text_processed)
        for word in stop_words:
            text_processed = re.sub(" " + word + " ", " ", text_processed)
        data_processed['text_processed'].append(text_processed)
    df = pd.DataFrame(data_processed)
    df.to_csv("./dataset/covid19_tweets_processed.csv")


def get_data(path):
    df = pd.read_csv(path, delimiter=",")
    tran = []
    try:
        for _, row in df.iterrows():
            tran.append(row['text_processed'].split())
    except Exception as e:
        print(row)
        raise e
    return tran


def write_result(f, item_set, rules):
    for num, se in item_set.items():
        for s in se:
            f.write(str(s) + "\n")
        f.write("=======================================\n")
    for item in rules:
        f.write(str(item[0]) + " -> " + str(item[1]) + ": " + str(item[2]) + "\n")


def main():
    from apriori_python import apriori
    if not os.path.exists('./dataset/covid19_tweets_processed.csv'):
        print("Running pre process data ...")
        data_path = './dataset/covid19_tweets.csv'
        pre_process(data_path)

    saved_data_path = './dataset/apriori/data.npy'
    if not os.path.exists(saved_data_path):
        print("Getting data ...")
        data_path = './dataset/covid19_tweets_processed.csv'
        transactions = asarray(get_data(data_path))
        save(saved_data_path, transactions)
    else:
        print("Load data ...")
        transactions = load(saved_data_path, allow_pickle=True)

    min_sup = 0.01
    min_conf = 0.01
    file_name = "./output/apriori/result_v2_" + str(min_sup) + "_" + str(min_conf) + ".txt"
    print("Running apriori ...")
    with open(file_name, "w") as f:
        f.write("Start time:"+str(datetime.now())+"\n")
        start_time = time.time()
        item_set, rules = apriori(transactions, minSup=min_sup, minConf=min_conf)
        elapsed_time = time.time() - start_time
        f.write("End time:" + str(datetime.now()) + "\n")
        f.write("Time execute: " + str(elapsed_time) + "\n")
        f.write("=======================================\n")
        print("Writing result ...")
        write_result(f, item_set, rules)


if __name__ == '__main__':
    main()
