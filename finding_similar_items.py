import pandas as pd
import re


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
    df = pd.read_csv(path)
    tran = []
    for _, row in df.iterrows():
        tran.append(row['text_processed'].split())
    return tran


def write_result(file_name, rules, min_length=2):
    with open(file_name, "w") as f:
        for item in rules:
            # first index of the inner list
            # Contains base item and add item
            pair = item[0]
            items = [x for x in pair]
            if len(items) < min_length:
                continue
            f.write("Rule: " + str(items[0]) + " -> " + str(items[1]) + " ("+", ".join(items)+")\n")

            # second index of the inner list
            f.write("Support: " + str(item[1]) + "\n")

            # third index of the list located at 0th
            # of the third index of the inner list

            f.write("Confidence: " + str(item[2][0][2]) + "\n")
            f.write("Lift: " + str(item[2][0][3]) + "\n")
            f.write("=====================================\n")


if __name__ == '__main__':
    from apyori import apriori
    data_path = './dataset/covid19_tweets.csv'
    pre_process(data_path)
    data_path = './dataset/covid19_tweets_processed.csv'
    transactions = get_data(data_path)
    result = apriori(transactions, min_support=0.001)
    list_result = list(result)
    write_result("./output/apriori/result_support_001.txt", result, min_length=2)
